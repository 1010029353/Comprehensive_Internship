"""
简化的图像描述生成推理脚本

核心功能：输入图片，输出描述
"""

import torch
import argparse
import pickle
import os
from PIL import Image
from torchvision import transforms
from build_vocab import Vocabulary
from model import ImageCaptioningModel


def load_image(image_path, transform, device):
    """
    加载和预处理图像
    
    Args:
        image_path (str): 图像文件路径
        transform: 图像变换
        device: 计算设备
        
    Returns:
        Tensor: 预处理后的图像张量
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image


def load_model_and_vocab(model_path, vocab_path, device, embed_size=512, attention_dim=512, decoder_dim=512, dropout=0.5):
    """
    加载模型和词汇表
    
    Args:
        model_path (str): 模型文件路径
        vocab_path (str): 词汇表文件路径
        device: 计算设备
        embed_size (int): 词嵌入维度
        attention_dim (int): 注意力维度
        decoder_dim (int): 解码器维度
        dropout (float): Dropout率
        
    Returns:
        tuple: (model, vocab)
    """
    # 加载词汇表
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # 创建模型（使用与训练时相同的参数）
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=embed_size,
        attention_dim=attention_dim,
        decoder_dim=decoder_dim,
        dropout=dropout
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'从检查点加载模型，轮次: {checkpoint.get("epoch", "未知")}')
    else:
        model.load_state_dict(checkpoint)
        print('直接加载模型状态字典')
    
    model.to(device)
    model.eval()
    
    return model, vocab


def decode_caption(sampled_ids, vocab):
    """
    将采样的词汇ID转换为文本描述
    
    Args:
        sampled_ids (Tensor): 采样的词汇ID
        vocab (Vocabulary): 词汇表
        
    Returns:
        str: 生成的描述文本
    """
    if torch.is_tensor(sampled_ids):
        sampled_ids = sampled_ids.cpu().numpy()
    
    if len(sampled_ids.shape) > 1:
        sampled_ids = sampled_ids[0]  # 取第一个样本
    
    # 转换ID为词汇
    caption_words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        caption_words.append(word)
    
    return ' '.join(caption_words)


def generate_caption(model, image_tensor, vocab, max_length=20):
    """
    生成图像描述
    
    Args:
        model: 训练好的模型
        image_tensor (Tensor): 预处理后的图像张量
        vocab (Vocabulary): 词汇表
        max_length (int): 最大生成长度
        
    Returns:
        str: 生成的描述
    """
    with torch.no_grad():
        sampled_ids, _ = model.sample(image_tensor, vocab, max_length=max_length, method='greedy')
    
    caption = decode_caption(sampled_ids, vocab)
    return caption


def main(args):
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载模型和词汇表
    print('正在加载模型和词汇表...')
    model, vocab = load_model_and_vocab(
        args.model_path, 
        args.vocab_path, 
        device,
        args.embed_size,
        args.attention_dim,
        args.hidden_size,
        args.dropout
    )
    print('模型加载完成')
    
    # 加载图像
    print(f'正在处理图像: {args.image_path}')
    image_tensor = load_image(args.image_path, transform, device)
    
    # 生成描述
    caption = generate_caption(model, image_tensor, vocab, args.max_length)
    
    # 输出结果
    print(f'\n生成的描述: {caption}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像描述生成推理脚本')

    parser.add_argument('--model_path', type=str, default='models/best_model.pth.tar',
                       help='训练好的模型文件路径')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                       help='词汇表文件路径')
    parser.add_argument('--image_path', type=str, required=True,
                       help='图像文件路径')
    parser.add_argument('--max_length', type=int, default=20,
                       help='最大生成长度')
    
    # 模型参数（应该与训练时相同）
    parser.add_argument('--embed_size', type=int, default=512,
                       help='词嵌入维度')
    parser.add_argument('--attention_dim', type=int, default=512,
                       help='注意力层维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='解码器隐状态维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout率')
    
    args = parser.parse_args()
    
    print('推理参数:')
    for key, value in vars(args).items():
        print(f'  {key}: {value}')
    print()
    
    main(args)