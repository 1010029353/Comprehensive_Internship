"""
改进的图像描述生成模型训练脚本

主要改进：
1. 使用注意力机制的编码器-解码器架构
2. 添加学习率调度和早停机制
3. 支持模型检查点保存和恢复
4. 添加验证集评估和可视化
5. 支持多种优化器和损失函数
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
import pickle
import time
import json
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("警告: 无法导入TensorBoard，将跳过日志记录")
        SummaryWriter = None

from data_loader import get_loader
from build_vocab import Vocabulary
from model import ImageCaptioningModel, AttentionEncoder, AttentionDecoder
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import matplotlib.pyplot as plt


class AverageMeter:
    """
    计算和存储平均值和当前值的工具类
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    计算top-k准确率
    
    Args:
        scores (Tensor): 模型预测分数 (N, vocab_size)
        targets (Tensor): 真实标签 (N,)
        k (int): top-k
        
    Returns:
        float: top-k准确率
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer, loss, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    保存模型检查点
    
    Args:
        epoch (int): 当前轮次
        model (nn.Module): 模型
        optimizer: 优化器
        loss (float): 当前损失
        is_best (bool): 是否是最佳模型
        checkpoint_dir (str): 检查点保存目录
        filename (str): 文件名
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth.tar')
        torch.save(state, best_path)
        print(f'最佳模型已保存至: {best_path}')


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    加载模型检查点
    
    Args:
        checkpoint_path (str): 检查点文件路径
        model (nn.Module): 模型
        optimizer: 优化器（可选）
        
    Returns:
        tuple: (start_epoch, best_loss)
    """
    if not os.path.exists(checkpoint_path):
        print(f'检查点文件不存在: {checkpoint_path}')
        return 0, float('inf')
    
    print(f'正在加载检查点: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    
    print(f'检查点加载成功，从第 {start_epoch} 轮开始训练')
    return start_epoch, best_loss


def validate(val_loader, model, criterion, device, vocab_size):
    """
    验证模型性能
    
    Args:
        val_loader: 验证数据加载器
        model: 模型
        criterion: 损失函数
        device: 设备
        vocab_size (int): 词汇表大小
        
    Returns:
        tuple: (loss, top5_accuracy)
    """
    model.eval()
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            
            # 前向传播
            predictions, caps_sorted, decode_lengths, alphas, sort_ind = model(images, captions, lengths)
            
            # 计算损失
            targets = caps_sorted[:, 1:]  # 去掉<start>标记
            
            # 移除填充部分
            predictions_copy = predictions.clone()
            for idx, length in enumerate(decode_lengths):
                predictions_copy[idx, length:, :] = 0
            
            # 计算损失
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
            
            loss = criterion(predictions, targets)
            
            # 计算准确率
            top5 = accuracy(predictions, targets, 5)
            
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            
            if i % 100 == 0:
                print(f'验证: [{i}/{len(val_loader)}] Loss: {losses.avg:.4f} Top-5 Accuracy: {top5accs.avg:.3f}%')
    
    print(f'验证结果 - Loss: {losses.avg:.4f}, Top-5 Accuracy: {top5accs.avg:.3f}%')
    return losses.avg, top5accs.avg


def train_epoch(train_loader, model, criterion, optimizer, epoch, device, vocab_size, 
                print_freq=100, grad_clip=5.0):
    """
    训练一个轮次
    
    Args:
        train_loader: 训练数据加载器
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        epoch (int): 当前轮次
        device: 设备
        vocab_size (int): 词汇表大小
        print_freq (int): 打印频率
        grad_clip (float): 梯度裁剪阈值
        
    Returns:
        float: 平均损失
    """
    model.train()
    
    losses = AverageMeter()
    top5accs = AverageMeter()
    
    start_time = time.time()
    
    for i, (images, captions, lengths) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        
        # 前向传播
        predictions, caps_sorted, decode_lengths, alphas, sort_ind = model(images, captions, lengths)
        
        # 计算损失
        targets = caps_sorted[:, 1:]  # 去掉<start>标记
        
        # 打包序列以便计算损失
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
        
        loss = criterion(predictions, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # 计算准确率
        top5 = accuracy(predictions, targets, 5)
        
        # 更新统计信息
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        
        # 打印训练信息
        if i % print_freq == 0:
            elapsed_time = time.time() - start_time
            print(f'轮次: [{epoch}][{i}/{len(train_loader)}] '
                  f'时间: {elapsed_time:.3f}s '
                  f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Top-5 Accuracy: {top5accs.val:.3f}% ({top5accs.avg:.3f}%) ')
    
    return losses.avg


def create_optimizer(model, args):
    """
    创建优化器
    
    Args:
        model: 模型
        args: 命令行参数
        
    Returns:
        optimizer: 优化器
    """
    # 分别设置编码器和解码器的学习率
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam([
            {'params': encoder_params, 'lr': args.encoder_lr},
            {'params': decoder_params, 'lr': args.decoder_lr}
        ], weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': args.encoder_lr},
            {'params': decoder_params, 'lr': args.decoder_lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD([
            {'params': encoder_params, 'lr': args.encoder_lr},
            {'params': decoder_params, 'lr': args.decoder_lr}
        ], momentum=0.9, weight_decay=args.weight_decay)
    
    return optimizer


def create_scheduler(optimizer, args):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        args: 命令行参数
        
    Returns:
        scheduler: 学习率调度器
    """
    if args.scheduler == 'step':
        return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=args.gamma, 
                               patience=args.patience)
    else:
        return None


def main(args):
    """
    主训练函数
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建输出目录
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    # 保存训练参数
    with open(os.path.join(args.model_path, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # 创建TensorBoard日志记录器
    writer = SummaryWriter(args.log_path) if SummaryWriter is not None else None
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载词汇表
    print('正在加载词汇表...')
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f'词汇表大小: {len(vocab)}')
    
    # 创建数据加载器
    print('正在创建训练数据加载器...')
    train_loader = get_loader(
        args.image_dir, args.caption_path, vocab, transform,
        args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    
    # 创建验证数据加载器（如果提供了验证集路径）
    val_loader = None
    if args.val_image_dir and args.val_caption_path:
        print('正在创建验证数据加载器...')
        val_transform = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        val_loader = get_loader(
            args.val_image_dir, args.val_caption_path, vocab, val_transform,
            args.batch_size, shuffle=False, num_workers=args.num_workers
        )
    
    # 创建模型
    print('正在创建模型...')
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_size,
        attention_dim=args.attention_dim,
        decoder_dim=args.hidden_size,
        dropout=args.dropout
    ).to(device)
    
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    # 加载检查点（如果存在）
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
    
    # 训练循环
    print('开始训练...')
    patience_counter = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'\n轮次 [{epoch}/{args.num_epochs}]')
        
        # 训练一个轮次
        train_loss = train_epoch(
            train_loader, model, criterion, optimizer, epoch, device,
            len(vocab), args.log_step, args.grad_clip
        )
        
        # 记录训练损失
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # 验证模型
        val_loss = float('inf')
        if val_loader is not None:
            val_loss, val_acc = validate(val_loader, model, criterion, device, len(vocab))
            if writer is not None:
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Accuracy/Top5', val_acc, epoch)
        
        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss if val_loader else train_loss)
            else:
                scheduler.step()
        
        # 记录学习率
        if writer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'LearningRate/Group{i}', param_group['lr'], epoch)
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % args.save_step == 0:
            save_checkpoint(
                epoch, model, optimizer, val_loss if val_loader else train_loss,
                is_best, args.model_path, f'checkpoint_epoch_{epoch+1}.pth.tar'
            )
        
        # 早停检查
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f'验证损失在 {args.early_stopping} 个轮次内没有改善，触发早停')
            break
        
        print(f'轮次 {epoch} 完成 - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
    
    # 保存最终模型
    save_checkpoint(
        epoch, model, optimizer, val_loss if val_loader else train_loss,
        False, args.model_path, 'final_model.pth.tar'
    )
    
    if writer is not None:
        writer.close()
    print('训练完成！')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练图像描述生成模型')
    
    # 数据路径参数
    parser.add_argument('--model_path', type=str, default='models/',
                       help='训练模型保存路径')
    parser.add_argument('--crop_size', type=int, default=224,
                       help='随机裁剪图像尺寸')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                       help='词汇表文件路径')
    parser.add_argument('--image_dir', type=str, default='data/resized2014',
                       help='训练图像目录')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                       help='训练标注文件路径')
    parser.add_argument('--val_image_dir', type=str, default='data/resizedval2014',
                       help='验证图像目录')
    parser.add_argument('--val_caption_path', type=str, default='data/annotations/captions_val2014.json',
                       help='验证标注文件路径')
    parser.add_argument('--log_path', type=str, default='logs/',
                       help='TensorBoard日志保存路径')
    
    # 训练参数
    parser.add_argument('--log_step', type=int, default=100,
                       help='打印日志的步数间隔')
    parser.add_argument('--save_step', type=int, default=1,
                       help='保存模型的步数间隔')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批量大小')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='数据加载工作线程数')
    
    # 模型参数
    parser.add_argument('--embed_size', type=int, default=512,
                       help='词嵌入维度')
    parser.add_argument('--attention_dim', type=int, default=512,
                       help='注意力层维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='解码器隐状态维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='优化器类型')
    parser.add_argument('--encoder_lr', type=float, default=1e-4,
                       help='编码器学习率')
    parser.add_argument('--decoder_lr', type=float, default=4e-4,
                       help='解码器学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                       help='梯度裁剪阈值')
    
    # 学习率调度参数
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['step', 'plateau', 'none'],
                       help='学习率调度器类型')
    parser.add_argument('--step_size', type=int, default=3,
                       help='StepLR的步长')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='学习率衰减因子')
    parser.add_argument('--patience', type=int, default=2,
                       help='ReduceLROnPlateau的耐心值')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default='',
                       help='恢复训练的检查点路径')
    parser.add_argument('--early_stopping', type=int, default=2,
                       help='早停轮次，0表示不使用早停')
    
    args = parser.parse_args()
    
    print('训练参数:')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    
    main(args)
