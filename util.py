
import torch
import math
import numpy as np

def lg10(x):
    return torch.div(torch.log(x), math.log(10))

def maxOfTwo(x, y):
    z = x.clone()
    maskYLarger = torch.lt(x, y)  #判断x是否小于y，小于为1不小于为0
    z[maskYLarger.detach()] = y[maskYLarger.detach()]
    return z  #z中与y不同的值为负值及大于y的

def nValid(x):
    return torch.sum(torch.eq(x, x).float())

def nNanElement(x):
    return torch.sum(torch.ne(x, x).float())

def getNanMask(x):
    return torch.ne(x, x)

def setNanToZero(input, target):
    #import ipdb;ipdb.set_trace()
    nanMask = getNanMask(target)  #判断target中是否有nan，nan所在的位置为1，非nan的位置为0，大小与target相同
    nValidElement = nValid(target)  #target中非nan即有效的像素数目
    
    zeroMask = torch.eq(target, 0)
    nValidElement = nValidElement - torch.sum(zeroMask)
    

    _input = input.clone()
    _target = target.clone()

    nanMask = (nanMask + zeroMask)>0
    #预测和gt中无效点的深度均置为0
    _input[nanMask] = 0
    _target[nanMask] = 0

    return _input, _target, nanMask, nValidElement


def evaluateError(output, target):
    #import ipdb;ipdb.set_trace()
    errors = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
              'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    
    #经过有效点筛选后的预测、gt、无效点为1有效点为0的标识矩阵、有效点像素数目
    _output, _target, nanMask, nValidElement = setNanToZero(output, target)

    if (nValidElement.data.cpu().numpy() > 0):
        diffMatrix = torch.abs(_output - _target)

        errors['MSE'] = torch.sum(torch.pow(diffMatrix, 2)) / nValidElement  #整个batch的平均，有效点预测和gt的差的平方求和取平均

        errors['MAE'] = torch.sum(diffMatrix) / nValidElement  #误差绝对值求和取平均

        realMatrix = torch.div(diffMatrix, _target)  #相对误差，误差除以gt，相同误差值在深度不同的点贡献不同（近的点更大）
        realMatrix[nanMask] = 0  #无效点的相对误差置为0
        errors['ABS_REL'] = torch.sum(realMatrix) / nValidElement  #相对误差求和取平均

        LG10Matrix = torch.abs(lg10(_output) - lg10(_target))  #预测和gt先取log10再做差，取绝对值，相同误差值在深度不同的点贡献不同（近的点更大）
        LG10Matrix[nanMask] = 0
        errors['LG10'] = torch.sum(LG10Matrix) / nValidElement

        yOverZ = torch.div(_output, _target)
        zOverY = torch.div(_target, _output)

        maxRatio = maxOfTwo(yOverZ, zOverY)  #预测值为正且小于gt的点赋值为zOverY

        errors['DELTA1'] = torch.sum(
            torch.le(maxRatio, 1.25).float()) / nValidElement
        errors['DELTA2'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 2)).float()) / nValidElement
        errors['DELTA3'] = torch.sum(
            torch.le(maxRatio, math.pow(1.25, 3)).float()) / nValidElement

        errors['MSE'] = float(errors['MSE'].data.cpu().numpy())
        errors['ABS_REL'] = float(errors['ABS_REL'].data.cpu().numpy())
        errors['LG10'] = float(errors['LG10'].data.cpu().numpy())
        errors['MAE'] = float(errors['MAE'].data.cpu().numpy())
        errors['DELTA1'] = float(errors['DELTA1'].data.cpu().numpy())
        errors['DELTA2'] = float(errors['DELTA2'].data.cpu().numpy())
        errors['DELTA3'] = float(errors['DELTA3'].data.cpu().numpy())

    return errors


def addErrors(errorSum, errors, batchSize):
    #import ipdb;ipdb.set_trace()
    errorSum['MSE']=errorSum['MSE'] + errors['MSE'] * batchSize
    errorSum['ABS_REL']=errorSum['ABS_REL'] + errors['ABS_REL'] * batchSize
    errorSum['LG10']=errorSum['LG10'] + errors['LG10'] * batchSize
    errorSum['MAE']=errorSum['MAE'] + errors['MAE'] * batchSize

    errorSum['DELTA1']=errorSum['DELTA1'] + errors['DELTA1'] * batchSize
    errorSum['DELTA2']=errorSum['DELTA2'] + errors['DELTA2'] * batchSize
    errorSum['DELTA3']=errorSum['DELTA3'] + errors['DELTA3'] * batchSize

    return errorSum


def averageErrors(errorSum, N):
    #import ipdb;ipdb.set_trace()
    averageError={'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                    'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    averageError['MSE'] = errorSum['MSE'] / N
    averageError['ABS_REL'] = errorSum['ABS_REL'] / N
    averageError['LG10'] = errorSum['LG10'] / N
    averageError['MAE'] = errorSum['MAE'] / N

    averageError['DELTA1'] = errorSum['DELTA1'] / N
    averageError['DELTA2'] = errorSum['DELTA2'] / N
    averageError['DELTA3'] = errorSum['DELTA3'] / N

    return averageError





	
