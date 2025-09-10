"""
图像描述生成模型词汇表构建工具

该模块用于从COCO数据集的标注文件中构建词汇表，用于图像描述生成任务。
主要功能包括：
1. 从JSON标注文件中提取图像描述文本
2. 对文本进行分词处理
3. 统计词频并过滤低频词
4. 构建词汇表映射关系
5. 保存词汇表到pickle文件
"""

import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary(object):
    """
    词汇表包装类
    
    用于管理词汇到索引的双向映射，支持词汇的添加和查询。
    包含特殊标记：<pad>（填充）、<start>（开始）、<end>（结束）、<unk>（未知词）
    """
    
    def __init__(self):
        """
        初始化词汇表
        
        创建词汇到索引和索引到词汇的双向映射字典
        """
        self.word2idx = {}  # 词汇到索引的映射
        self.idx2word = {}  # 索引到词汇的映射
        self.idx = 0        # 当前索引计数器
    
    def add_word(self, word):
        """
        向词汇表中添加新词汇
        
        Args:
            word (str): 要添加的词汇
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        """
        获取词汇对应的索引
        
        Args:
            word (str): 查询的词汇
            
        Returns:
            int: 词汇对应的索引，如果词汇不存在则返回<unk>标记的索引
        """
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        """
        返回词汇表的大小
        
        Returns:
            int: 词汇表中词汇的总数
        """
        return len(self.word2idx)


def load_and_process_captions(json_path):
    """
    加载并处理COCO数据集的标注文件
    
    Args:
        json_path (str): COCO标注文件的路径
        
    Returns:
        tuple: (COCO对象, 标注ID列表)
    """
    print(f"正在加载COCO数据集标注文件: {json_path}")
    coco = COCO(json_path)
    ids = list(coco.anns.keys())
    print(f"成功加载 {len(ids)} 条图像描述标注")
    return coco, ids


def tokenize_captions(coco, ids):
    """
    对所有图像描述进行分词处理
    
    Args:
        coco (COCO): COCO数据集对象
        ids (list): 标注ID列表
        
    Returns:
        Counter: 词频统计计数器
    """
    print("开始对图像描述进行分词处理...")
    counter = Counter()
    
    for i, annotation_id in enumerate(ids):
        # 获取图像描述文本
        caption = str(coco.anns[annotation_id]['caption'])
        
        # 转换为小写并进行分词
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        
        # 更新词频统计
        counter.update(tokens)
        
        # 每处理1000条描述输出一次进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 [{i+1}/{len(ids)}] 条图像描述")
    
    print("分词处理完成！")
    return counter


def filter_words_by_frequency(counter, threshold):
    """
    根据词频阈值过滤词汇
    
    Args:
        counter (Counter): 词频统计计数器
        threshold (int): 词频阈值，低于此阈值的词汇将被过滤
        
    Returns:
        list: 过滤后的词汇列表
    """
    print(f"正在过滤词频低于 {threshold} 的词汇...")
    
    # 过滤低频词汇
    filtered_words = [word for word, count in counter.items() if count >= threshold]
    
    print(f"过滤前词汇总数: {len(counter)}")
    print(f"过滤后词汇总数: {len(filtered_words)}")
    
    return filtered_words


def create_vocabulary(words):
    """
    创建词汇表并添加特殊标记
    
    Args:
        words (list): 词汇列表
        
    Returns:
        Vocabulary: 构建完成的词汇表对象
    """
    print("正在创建词汇表...")
    
    # 创建词汇表对象
    vocab = Vocabulary()
    
    # 添加特殊标记（顺序很重要）
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
    for token in special_tokens:
        vocab.add_word(token)
    
    # 添加过滤后的词汇
    for word in words:
        vocab.add_word(word)
    
    print(f"词汇表创建完成，总词汇数: {len(vocab)}")
    return vocab


def build_vocab(json_path, threshold):
    """
    构建词汇表的主要流程
    
    Args:
        json_path (str): COCO标注文件路径
        threshold (int): 词频阈值
        
    Returns:
        Vocabulary: 构建完成的词汇表对象
    """
    # 1. 加载和处理标注文件
    coco, ids = load_and_process_captions(json_path)
    
    # 2. 分词处理
    counter = tokenize_captions(coco, ids)
    
    # 3. 根据词频过滤词汇
    words = filter_words_by_frequency(counter, threshold)
    
    # 4. 创建词汇表
    vocab = create_vocabulary(words)
    
    return vocab


def save_vocabulary(vocab, vocab_path):
    """
    保存词汇表到文件
    
    Args:
        vocab (Vocabulary): 词汇表对象
        vocab_path (str): 保存路径
    """
    print(f"正在保存词汇表到: {vocab_path}")
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"词汇表已成功保存！")
    print(f"词汇表大小: {len(vocab)}")
    print(f"保存路径: {vocab_path}")


def main(args):
    """
    主函数：执行词汇表构建的完整流程
    
    Args:
        args: 命令行参数对象
    """
    print("=" * 50)
    print("开始构建图像描述生成模型词汇表")
    print("=" * 50)
    
    # 构建词汇表
    vocab = build_vocab(json_path=args.caption_path, threshold=args.threshold)
    
    # 保存词汇表
    save_vocabulary(vocab, args.vocab_path)
    
    print("=" * 50)
    print("词汇表构建完成！")
    print("=" * 50)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='从COCO数据集构建图像描述生成模型的词汇表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python build_vocab.py --caption_path data/annotations/captions_train2014.json --threshold 4
  python build_vocab.py --vocab_path ./data/my_vocab.pkl --threshold 5
        """
    )
    
    parser.add_argument(
        '--caption_path', 
        type=str,
        default='data/annotations/captions_train2014.json',
        help='COCO训练集标注文件的路径 (默认: data/annotations/captions_train2014.json)'
    )
    
    parser.add_argument(
        '--vocab_path', 
        type=str, 
        default='./data/vocab.pkl',
        help='词汇表保存路径 (默认: ./data/vocab.pkl)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=int, 
        default=4,
        help='词频阈值，低于此值的词汇将被过滤 (默认: 4)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 执行主程序
    main(args)