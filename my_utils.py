import torch
import torch.nn.functional as F

def compute_depth_loss(depth_map):
    """
    计算深度图的平滑损失
    """
    dx = torch.abs(depth_map[:, :, 1:] - depth_map[:, :, :-1])
    dy = torch.abs(depth_map[:, 1:, :] - depth_map[:, :-1, :])
    return torch.mean(dx) + torch.mean(dy)

def sobel_loss(pred, target):
    """
    计算预测图像和真实图像的 Sobel 梯度损失
    """
    # 定义Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float).to(pred.device)
    
    # 适配通道 (假设是RGB 3通道)
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1) 
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    
    # 确保输入是 (B, C, H, W)
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
    # 计算梯度
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=3)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=3)
    target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=3)
    target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=3)
    
    pred_grad = torch.abs(pred_grad_x) + torch.abs(pred_grad_y)
    target_grad = torch.abs(target_grad_x) + torch.abs(target_grad_y)
    
    # 返回梯度的L1距离
    return F.l1_loss(pred_grad, target_grad)

def fft_loss(pred, target):
    """
    计算预测图像和真实图像在频域的 L1 损失
    """
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # 1. 转换到频域
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')

    # 2. 计算频域的 L1 距离
    loss = torch.abs(pred_fft - target_fft).mean()
    
    return loss