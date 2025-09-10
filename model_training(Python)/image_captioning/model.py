"""
改进的图像描述生成模型

基于注意力机制的编码器-解码器架构，主要特点：
1. 使用ResNet作为特征提取器，保留空间信息
2. 实现加性注意力机制，让模型关注图像的不同区域
3. 结合LSTM的序列解码器，支持上下文信息传递
4. 支持束搜索(Beam Search)生成更好的描述
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class AttentionEncoder(nn.Module):
    """
    基于注意力机制的图像编码器
    
    使用预训练的ResNet提取图像特征，保留空间维度信息
    输出特征图可以用于后续的注意力计算
    """
    
    def __init__(self, encoded_image_size=14):
        """
        初始化编码器
        
        Args:
            encoded_image_size (int): 编码后特征图的空间尺寸
        """
        super(AttentionEncoder, self).__init__()
        self.enc_image_size = encoded_image_size
        
        # 使用预训练的ResNet-101作为特征提取器
        resnet = models.resnet101(pretrained=True)
        
        # 移除最后的平均池化层和全连接层，保留空间信息
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
        # 自适应池化层，确保输出尺寸固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        
        # 特征维度调整层
        self.fine_tune()
        
    def forward(self, images):
        """
        前向传播
        
        Args:
            images (Tensor): 输入图像 (batch_size, 3, image_size, image_size)
            
        Returns:
            Tensor: 编码后的特征图 (batch_size, 2048, enc_image_size, enc_image_size)
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, enc_image_size, enc_image_size)
        
        # 重新排列维度以便后续处理
        out = out.permute(0, 2, 3, 1)  # (batch_size, enc_image_size, enc_image_size, 2048)
        
        return out
    
    def fine_tune(self, fine_tune=True):
        """
        控制是否对预训练参数进行微调
        
        Args:
            fine_tune (bool): 是否允许微调
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        # 只微调后面几层
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    加性注意力机制
    
    计算解码器隐状态与编码器特征图之间的注意力权重
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        初始化注意力层
        
        Args:
            encoder_dim (int): 编码器特征维度
            decoder_dim (int): 解码器隐状态维度  
            attention_dim (int): 注意力层的隐藏维度
        """
        super(Attention, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # 编码器特征线性变换
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # 解码器状态线性变换
        self.full_att = nn.Linear(attention_dim, 1)  # 最终注意力分数计算
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 在空间维度上进行softmax
        
    def forward(self, encoder_out, decoder_hidden):
        """
        计算注意力权重和上下文向量
        
        Args:
            encoder_out (Tensor): 编码器输出 (batch_size, num_pixels, encoder_dim)
            decoder_hidden (Tensor): 解码器隐状态 (batch_size, decoder_dim)
            
        Returns:
            tuple: (attention_weighted_encoding, alpha)
                - attention_weighted_encoding: 加权后的上下文向量
                - alpha: 注意力权重
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        
        # 扩展解码器状态以匹配空间维度
        att2 = att2.unsqueeze(1)  # (batch_size, 1, attention_dim)
        
        # 计算注意力能量
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch_size, num_pixels)
        
        # 计算注意力权重
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        
        # 计算加权上下文向量
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        return attention_weighted_encoding, alpha


class AttentionDecoder(nn.Module):
    """
    基于注意力机制的解码器
    
    使用LSTM生成图像描述，在每个时间步计算注意力权重
    """
    
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        初始化解码器
        
        Args:
            attention_dim (int): 注意力层维度
            embed_dim (int): 词嵌入维度
            decoder_dim (int): 解码器隐状态维度
            vocab_size (int): 词汇表大小
            encoder_dim (int): 编码器特征维度
            dropout (float): Dropout率
        """
        super(AttentionDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        # 注意力机制
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # LSTM解码器
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        
        # 初始化隐状态和细胞状态的线性层
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # 门控机制，用于控制注意力信息的流入
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 输出层
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """
        初始化模型权重
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
    def load_pretrained_embeddings(self, embeddings):
        """
        加载预训练的词嵌入
        
        Args:
            embeddings (Tensor): 预训练的词嵌入矩阵
        """
        self.embedding.weight = nn.Parameter(embeddings)
        
    def fine_tune_embeddings(self, fine_tune=True):
        """
        控制是否微调词嵌入
        
        Args:
            fine_tune (bool): 是否允许微调
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
            
    def init_hidden_state(self, encoder_out):
        """
        根据编码器输出初始化LSTM的隐状态
        
        Args:
            encoder_out (Tensor): 编码器输出
            
        Returns:
            tuple: (h, c) LSTM的初始隐状态和细胞状态
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        训练时的前向传播
        
        Args:
            encoder_out (Tensor): 编码器输出 (batch_size, enc_image_size, enc_image_size, encoder_dim)
            encoded_captions (Tensor): 编码后的描述 (batch_size, max_caption_length)
            caption_lengths (list): 每个描述的实际长度
            
        Returns:
            tuple: (predictions, encoded_captions, decode_lengths, alphas, sort_ind)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # 展平编码器输出的空间维度
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # 将长度列表转换为张量并排序（有助于pack_padded_sequence）
        caption_lengths = torch.tensor(caption_lengths)
        caption_lengths, sort_ind = caption_lengths.sort(0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # 词嵌入
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        # 初始化LSTM状态
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        # 我们不会在<end>标记处解码，所以有效长度减1
        decode_lengths = (caption_lengths - 1).tolist()
        
        # 创建用于存储预测和注意力权重的张量
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        
        # 在每个时间步进行解码
        for t in range(max(decode_lengths)):
            # 找到在此时间步仍在解码的批次
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # 计算注意力权重和上下文向量
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            
            # 门控注意力
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # 门控权重
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM解码步骤
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            # 计算词汇分布
            preds = self.fc(self.dropout_layer(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
    def sample(self, encoder_out, max_length=20, vocab=None):
        """
        推理时的采样生成
        
        Args:
            encoder_out (Tensor): 编码器输出
            max_length (int): 最大生成长度
            vocab (Vocabulary): 词汇表对象
            
        Returns:
            tuple: (sampled_caption, alphas)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # 展平编码器输出
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # 初始化
        h, c = self.init_hidden_state(encoder_out)
        
        # 存储结果
        sampled_ids = []
        alphas = []
        
        # 开始标记
        inputs = torch.tensor([vocab('<start>')]).to(encoder_out.device)
        inputs = inputs.expand(batch_size)  # (batch_size,)
        
        for t in range(max_length):
            # 词嵌入
            embeddings = self.embedding(inputs).squeeze(1)  # (batch_size, embed_dim)
            
            # 注意力计算
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            alphas.append(alpha.cpu())
            
            # 门控注意力
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM解码
            h, c = self.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))
            
            # 预测下一个词
            preds = self.fc(h)  # (batch_size, vocab_size)
            predicted = preds.argmax(dim=1)  # (batch_size,)
            
            sampled_ids.append(predicted.cpu())
            inputs = predicted
            
            # 如果预测到结束标记则停止
            if vocab and predicted.item() == vocab('<end>'):
                break
                
        # 转换为张量
        sampled_ids = torch.stack(sampled_ids, dim=1)  # (batch_size, length)
        alphas = torch.stack(alphas, dim=1)  # (batch_size, length, num_pixels)
        
        return sampled_ids, alphas


def beam_search(decoder, encoder_out, vocab, beam_size=3, max_length=20):
    """
    束搜索解码算法
    
    Args:
        decoder (AttentionDecoder): 解码器模型
        encoder_out (Tensor): 编码器输出
        vocab (Vocabulary): 词汇表
        beam_size (int): 束大小
        max_length (int): 最大长度
        
    Returns:
        list: 生成的描述序列
    """
    k = beam_size
    encoder_dim = encoder_out.size(-1)
    
    # 展平编码器输出
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
    
    # 初始化
    k_prev_words = torch.tensor([[vocab('<start>')]] * k).to(encoder_out.device)  # (k, 1)
    seqs = k_prev_words  # (k, 1)
    top_k_scores = torch.zeros(k, 1).to(encoder_out.device)  # (k, 1)
    
    # 完成的序列列表
    complete_seqs = []
    complete_seqs_scores = []
    
    # 初始化LSTM状态
    h, c = decoder.init_hidden_state(encoder_out)  # (k, decoder_dim)
    
    # 开始解码
    step = 1
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (k, embed_dim)
        
        # 注意力计算
        attention_weighted_encoding, alpha = decoder.attention(encoder_out, h)
        
        # 门控注意力
        gate = decoder.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        
        # LSTM解码
        h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))
        
        # 预测
        scores = decoder.fc(h)  # (k, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        
        # 计算累积分数
        scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)
        
        # 第一步时，所有候选都来自第一个序列
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (k)
        else:
            # 展开所有可能的下一个词
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (k)
        
        # 转换为2D坐标
        vocab_size = decoder.vocab_size
        prev_word_inds = top_k_words // vocab_size  # (k)
        next_word_inds = top_k_words % vocab_size  # (k)
        
        # 添加新词到序列
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (k, step+1)
        
        # 检查哪些序列完成了
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != vocab('<end>')]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        
        # 设置aside完成的序列
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # 减少束大小
        
        # 如果k变为0或达到最大长度，则停止
        if k == 0 or step == max_length:
            break
            
        # 继续处理未完成的序列
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        
        step += 1
    
    # 选择最佳序列
    if len(complete_seqs_scores) > 0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
    else:
        seq = seqs[0].tolist()
        
    return seq


class ImageCaptioningModel(nn.Module):
    """
    完整的图像描述生成模型
    
    结合编码器和解码器的完整模型
    """
    
    def __init__(self, vocab_size, embed_dim=512, attention_dim=512, decoder_dim=512, 
                 encoder_dim=2048, dropout=0.5):
        """
        初始化完整模型
        
        Args:
            vocab_size (int): 词汇表大小
            embed_dim (int): 词嵌入维度
            attention_dim (int): 注意力维度
            decoder_dim (int): 解码器维度
            encoder_dim (int): 编码器维度
            dropout (float): Dropout率
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = AttentionEncoder()
        self.decoder = AttentionDecoder(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout
        )
        
    def forward(self, images, captions, caption_lengths):
        """
        前向传播
        
        Args:
            images (Tensor): 输入图像
            captions (Tensor): 描述文本
            caption_lengths (Tensor): 描述长度
            
        Returns:
            解码器的输出
        """
        encoder_out = self.encoder(images)
        return self.decoder(encoder_out, captions, caption_lengths)
    
    def sample(self, images, vocab, max_length=20, method='greedy'):
        """
        生成图像描述
        
        Args:
            images (Tensor): 输入图像
            vocab (Vocabulary): 词汇表
            max_length (int): 最大长度
            method (str): 生成方法 ('greedy' 或 'beam')
            
        Returns:
            生成的描述
        """
        encoder_out = self.encoder(images)
        
        if method == 'beam':
            return beam_search(self.decoder, encoder_out, vocab, max_length=max_length)
        else:
            return self.decoder.sample(encoder_out, max_length=max_length, vocab=vocab)
