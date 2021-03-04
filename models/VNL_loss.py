import torch
import torch.nn as nn
import numpy as np


class VNL_Loss(nn.Module):
    """
    Virtual Normal Loss Function.
    """
    def __init__(self, input_size, focal_x=519, focal_y=519, 
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=0.0001, sample_ratio=0.15):
        super(VNL_Loss, self).__init__()
        self.fx = torch.tensor([focal_x], dtype=torch.float32).cuda()  #相机x方向焦距，与数据集有关
        self.fy = torch.tensor([focal_y], dtype=torch.float32).cuda()  #相机y方向焦距
        self.input_size = input_size  #输入变量的大小
        self.u0 = torch.tensor(input_size[1] // 2, dtype=torch.float32).cuda()  #相机光学中心，用输入变量的中心来近似，调用损失时输入图像的大小决定
        self.v0 = torch.tensor(input_size[0] // 2, dtype=torch.float32).cuda()
        self.init_image_coor()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio

    def init_image_coor(self):
        x_row = np.arange(0, self.input_size[1])  #[0,1,2,...,self.input_size[1]-1]
        x = np.tile(x_row, (self.input_size[0], 1))  #把x_row在列上复制self.input_size[0]次，行上复制1次
        x = x[np.newaxis, :, :]  #[1,self.input_size[0], self.input_size[1]]
        x = x.astype(np.float32)
        x = torch.from_numpy(x.copy()).cuda()
        self.u_u0 = x #- self.u0  #输入图像上每个点的二维横坐标减去光学中心横坐标

        y_col = np.arange(0, self.input_size[0])  # y_col = np.arange(0, height)
        y = np.tile(y_col, (self.input_size[1], 1)).T  #[1,self.input_size[0], self.input_size[1]]
        y = y[np.newaxis, :, :]
        y = y.astype(np.float32)
        y = torch.from_numpy(y.copy()).cuda()
        self.v_v0 = y #- self.v0  #输入图像上每个点的而且纵坐标减去光学中心纵坐标

    def transfer_xyz(self, depth):
        ##利用深度图构造三维点云坐标
        #import ipdb;ipdb.set_trace()
        x = self.u_u0.repeat((depth.shape[0],1,1,1)) #* torch.abs(depth) / self.fx
        y = self.v_v0.repeat((depth.shape[0],1,1,1))  #* torch.abs(depth) / self.fy
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]，c=3
        return pw

    def select_index(self):
        valid_width = self.input_size[1]
        valid_height = self.input_size[0]
        num = valid_width * valid_height  #输入图像像素点数目
        p1 = np.random.choice(num, int(num * self.sample_ratio), replace=True)  #在num中随机选择num * self.sample_ratio个数字
        np.random.shuffle(p1)
        p2 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p2)
        p3 = np.random.choice(num, int(num * self.sample_ratio), replace=True)
        np.random.shuffle(p3)

        p1_x = p1 % self.input_size[1]
        p1_y = (p1 / self.input_size[1]).astype(np.int)  #p1随机选择的像素点的坐标

        p2_x = p2 % self.input_size[1]
        p2_y = (p2 / self.input_size[1]).astype(np.int)

        p3_x = p3 % self.input_size[1]
        p3_y = (p3 / self.input_size[1]).astype(np.int)
        
        ##选择了3组，每组num * self.sample_ratio个点，返回每个点的x,y坐标
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']

        pw1 = pw[:, p1_y, p1_x, :] #第一组点在整个batch上的三维坐标（x,y,z）
        pw2 = pw[:, p2_y, p2_x, :]
        pw3 = pw[:, p3_y, p3_x, :]
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.cat([pw1[:, :, :, np.newaxis], pw2[:, :, :, np.newaxis], pw3[:, :, :, np.newaxis]], 3)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        #用3组点构造向量
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]  #第一组向量的（x,y,z）,[B, num * self.sample_ratio, 3(x,y,z)]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ###ignore linear
        #每张图上选择num * self.sample_ratio个虚拟法线，每个向量表示为(x,y,z),每个虚拟法线表示为3个向量，
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1)  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.view(m_batchsize * groups, -1, index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2) #每个向量的2范数，[bn,3,1]
        nm = torch.bmm(q_norm.view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index)) #矩阵乘法，[bn,3,3]
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)],向量内积
        norm_energy = energy / (nm + 1e-8) #余弦，
        norm_energy = norm_energy.view(m_batchsize * groups, -1) #[bn,9],9表示向量1分别与向量1,2,3的余弦，2分别与1,2,3的余弦，3分别与1,2,3的余弦
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3  # igonre，3为1与1,2与2,3与3的夹角余弦值的和
        mask_cos = mask_cos.view(m_batchsize, groups)  #在选择的groups中不满足条件要被忽略的那些
        ##ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3  #一组点的三个向量的深度值都大于0.001

        ###ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0  #至少有1个向量的x小于阈值即向量的起始点在x上离得近
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0  #[b,n]
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw  #mask中1为有效的组

    def select_points_groups(self, gt_depth, pred_depth):
        pw_gt = self.transfer_xyz(gt_depth)  #gt的三维点云坐标
        pw_pred = self.transfer_xyz(pred_depth)  #pred的三维点云坐标，[B,H,W,C],C=(x,y,z)
        B, C, H, W = gt_depth.shape
        p123 = self.select_index()  #字典，6个key，分别为第1,2,3组点的x，y坐标，每组包括num * self.sample_ratio个点
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        ##mask中1表示有效选择的这组点，可用于后续计算
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt,
                                              delta_cos=0.867,
                                              delta_diff_x=0.005,
                                              delta_diff_y=0.005,
                                              delta_diff_z=0.005)

        # pred的3维点云坐标[b, n, 3（x,y,z）, 3(p1,p2,p3)]，
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001  #预测深度为0的点深度赋值为0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2) #[b*1,n*9]→[b,3,3,n]→[b,n,3,3]
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, gt_depth, pred_depth, select=True):
        """
        Virtual normal loss.
        :param pred_depth: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        gt_points, dt_points = self.select_points_groups(gt_depth, pred_depth)  #有效可用于计算的点

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]  #gt中每组有效点构成的向量
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        dt_p12 = dt_points[:, :, :, 1] - dt_points[:, :, :, 0]
        dt_p13 = dt_points[:, :, :, 2] - dt_points[:, :, :, 0]
        
        #求法线
        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        dt_normal = torch.cross(dt_p12, dt_p13, dim=2)
        dt_norm = torch.norm(dt_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        dt_mask = dt_norm == 0.0
        gt_mask = gt_norm == 0.0
        dt_mask = dt_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        dt_mask *= 0.01
        gt_mask *= 0.01
        gt_norm = gt_norm + gt_mask
        dt_norm = dt_norm + dt_mask
        gt_normal = gt_normal / gt_norm
        dt_normal = dt_normal / dt_norm
        
        loss = torch.abs(gt_normal - dt_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    import cv2
    vnl_loss = VNL_Loss(1.0, 1.0, (480, 640))
    pred_depth = np.ones([2, 1, 480, 640])
    gt_depth = np.ones([2, 1, 480, 640])
    gt_depth = torch.tensor(np.asarray(gt_depth, np.float32)).cuda()
    pred_depth = torch.tensor(np.asarray(pred_depth, np.float32)).cuda()
    loss = vnl_loss.cal_VNL_loss(pred_depth, gt_depth)
    print(loss)
