"""
COCO数据集加载器

该模块用于加载和处理COCO数据集，为图像描述生成任务提供数据支持。
主要功能包括：
1. COCO数据集的自定义Dataset类实现
2. 图像和描述文本的预处理
3. 批量数据处理和填充
4. 数据加载器的创建和配置

支持PyTorch的DataLoader接口，可以高效地进行批量数据加载和预处理。
"""

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)

# Also download punkt for compatibility
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class CocoDataset(data.Dataset):
    """
    COCO数据集自定义类
    
    继承自PyTorch的Dataset类，用于加载COCO数据集的图像和对应的描述文本。
    支持图像变换和文本标记化处理。
    """
    
    def __init__(self, root, json, vocab, transform=None):
        """
        初始化COCO数据集
        
        Args:
            root (str): 图像文件根目录路径
            json (str): COCO标注文件路径
            vocab (Vocabulary): 词汇表对象
            transform (callable, optional): 图像变换函数
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        
        print(f"COCO数据集初始化完成")
        print(f"图像目录: {root}")
        print(f"标注文件: {json}")
        print(f"数据样本数量: {len(self.ids)}")
    
    def _load_image(self, image_id):
        """
        加载单张图像
        
        Args:
            image_id (int): 图像ID
            
        Returns:
            PIL.Image: 加载的RGB图像
        """
        # 获取图像文件名
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root, image_info['file_name'])
        
        # 加载图像并转换为RGB格式
        image = Image.open(image_path).convert('RGB')
        
        return image
    
    def _process_caption(self, caption_text):
        """
        处理描述文本，将其转换为词汇索引序列
        
        Args:
            caption_text (str): 原始描述文本
            
        Returns:
            list: 词汇索引列表，包含开始和结束标记
        """
        # 将描述文本转换为小写并进行分词
        tokens = nltk.tokenize.word_tokenize(str(caption_text).lower())
        
        # 构建词汇索引序列
        caption_indices = []
        caption_indices.append(self.vocab('<start>'))  # 添加开始标记
        caption_indices.extend([self.vocab(token) for token in tokens])  # 添加词汇索引
        caption_indices.append(self.vocab('<end>'))    # 添加结束标记
        
        return caption_indices
    
    def __getitem__(self, index):
        """
        获取单个数据样本
        
        Args:
            index (int): 数据索引
            
        Returns:
            tuple: (image, target)
                - image: 经过变换的图像张量
                - target: 描述文本对应的词汇索引张量
        """
        # 获取标注信息
        ann_id = self.ids[index]
        annotation = self.coco.anns[ann_id]
        caption_text = annotation['caption']
        image_id = annotation['image_id']
        
        # 加载和处理图像
        image = self._load_image(image_id)
        if self.transform is not None:
            image = self.transform(image)
        
        # 处理描述文本
        caption_indices = self._process_caption(caption_text)
        target = torch.Tensor(caption_indices)
        
        return image, target
    
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 数据集中样本的总数
        """
        return len(self.ids)


def create_padded_batch(captions):
    """
    创建填充后的批量描述数据
    
    Args:
        captions (list): 描述文本列表，每个元素是一个张量
        
    Returns:
        tuple: (targets, lengths)
            - targets: 填充后的二维张量 (batch_size, max_length)
            - lengths: 每个描述的实际长度列表
    """
    # 获取每个描述的长度
    lengths = [len(cap) for cap in captions]
    max_length = max(lengths)
    
    # 创建填充张量，使用0进行填充
    targets = torch.zeros(len(captions), max_length).long()
    
    # 填充每个描述
    for i, cap in enumerate(captions):
        end_idx = lengths[i]
        targets[i, :end_idx] = cap[:end_idx]
    
    return targets, lengths


def collate_fn(data):
    """
    自定义批量数据整理函数
    
    由于描述文本长度不一致，需要自定义整理函数来处理填充。
    该函数将按描述长度降序排列数据，然后进行填充处理。
    
    Args:
        data (list): 数据样本列表，每个元素是(image, caption)元组
            - image: 形状为(3, 256, 256)的图像张量
            - caption: 长度可变的描述张量
    
    Returns:
        tuple: (images, targets, lengths)
            - images: 批量图像张量，形状为(batch_size, 3, 256, 256)
            - targets: 填充后的描述张量，形状为(batch_size, padded_length)
            - lengths: 每个描述的有效长度列表
    """
    # 按描述长度降序排列（有助于RNN训练效率）
    data.sort(key=lambda x: len(x[1]), reverse=True)
    
    # 分离图像和描述
    images, captions = zip(*data)
    
    # 将图像元组堆叠为4D张量
    images = torch.stack(images, 0)
    
    # 创建填充后的描述批量数据
    targets, lengths = create_padded_batch(captions)
    
    return images, targets, lengths


def create_data_transforms(image_size=256):
    """
    创建图像数据变换
    
    Args:
        image_size (int): 目标图像尺寸
        
    Returns:
        torchvision.transforms.Compose: 图像变换组合
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """
    创建COCO数据集的数据加载器
    
    Args:
        root (str): 图像文件根目录路径
        json (str): COCO标注文件路径
        vocab (Vocabulary): 词汇表对象
        transform (callable): 图像变换函数
        batch_size (int): 批量大小
        shuffle (bool): 是否随机打乱数据
        num_workers (int): 数据加载的工作线程数
        
    Returns:
        torch.utils.data.DataLoader: 配置好的数据加载器
            每次迭代返回(images, captions, lengths)：
            - images: 形状为(batch_size, 3, 224, 224)的图像张量
            - captions: 形状为(batch_size, padded_length)的描述张量
            - lengths: 长度为batch_size的有效长度列表
    """
    print("正在创建COCO数据加载器...")
    
    # 创建COCO数据集实例
    coco_dataset = CocoDataset(
        root=root,
        json=json,
        vocab=vocab,
        transform=transform
    )
    
    # 创建数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset=coco_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()  # 如果有GPU则启用内存固定
    )
    
    print(f"数据加载器创建完成")
    print(f"批量大小: {batch_size}")
    print(f"工作线程数: {num_workers}")
    print(f"数据打乱: {shuffle}")
    
    return data_loader


def load_vocabulary(vocab_path):
    """
    加载词汇表文件
    
    Args:
        vocab_path (str): 词汇表文件路径
        
    Returns:
        Vocabulary: 加载的词汇表对象
    """
    print(f"正在加载词汇表: {vocab_path}")
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"词汇表加载完成，词汇数量: {len(vocab)}")
    return vocab


def get_coco_data_loader(image_dir, caption_file, vocab_path, 
                        batch_size=32, image_size=256, shuffle=True, num_workers=4):
    """
    便捷函数：一步创建完整的COCO数据加载器
    
    Args:
        image_dir (str): 图像目录路径
        caption_file (str): 标注文件路径
        vocab_path (str): 词汇表文件路径
        batch_size (int): 批量大小
        image_size (int): 图像尺寸
        shuffle (bool): 是否打乱数据
        num_workers (int): 工作线程数
        
    Returns:
        torch.utils.data.DataLoader: 配置好的数据加载器
    """
    # 加载词汇表
    vocab = load_vocabulary(vocab_path)
    
    # 创建图像变换
    transform = create_data_transforms(image_size)
    
    # 创建数据加载器
    data_loader = get_loader(
        root=image_dir,
        json=caption_file,
        vocab=vocab,
        transform=transform,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return data_loader