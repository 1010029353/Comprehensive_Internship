"""
模型转换脚本：将PyTorch图像描述模型转为TorchScript格式，并导出词汇表JSON

@author SaltyHeart

这个版本修复了trace的动态流问题，用script替换；简化了模块，只封装必要推理逻辑；硬编码特殊token ID以兼容C++。
"""

import torch
import argparse
import pickle
import json
import typing
from model import ImageCaptioningModel
from build_vocab import Vocabulary

class InferenceModule(torch.nn.Module):
    # 推理模块：封装模型的采样逻辑，便于转换为TorchScript
    def __init__(self, model: ImageCaptioningModel):
        super(InferenceModule, self).__init__()
        self.encoder = model.encoder  # 图像编码器
        self.embedding = model.decoder.embedding  # 词嵌入层
        self.attention = model.decoder.attention  # 注意力机制
        self.decode_step = model.decoder.decode_step  # LSTM解码步骤
        self.f_beta = model.decoder.f_beta  # 门控线性层
        self.sigmoid = model.decoder.sigmoid  # sigmoid激活
        self.fc = model.decoder.fc  # 输出全连接层
        self.init_h = model.decoder.init_h  # 初始隐状态线性层
        self.init_c = model.decoder.init_c  # 初始细胞状态线性层
        self.encoder_dim: int = model.decoder.encoder_dim  # 编码维度
        self.decoder_dim: int = model.decoder.decoder_dim  # 解码维度

    @torch.jit.export
    def init_hidden_state(self, encoder_out: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        # 根据编码输出计算初始LSTM状态
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    @torch.jit.export
    def forward(self, image: torch.Tensor, max_length: int) -> torch.Tensor:
        # 编码输入图像
        encoder_out = self.encoder(image)
        batch_size: int = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)

        # 初始化LSTM状态
        h, c = self.init_hidden_state(encoder_out)

        # 准备采样ID列表
        sampled_ids = torch.jit.annotate(typing.List[torch.Tensor], [])

        # 起始输入：嵌入<start> token (ID=1)
        inputs = self.embedding(torch.full((batch_size,), 1, dtype=torch.int64, device=image.device))

        # 循环生成序列
        for t in range(max_length):
            # 计算注意力权重和上下文
            awe, alpha = self.attention(encoder_out, h)

            # 应用门控机制
            gate = self.sigmoid(self.f_beta(h))
            awe = gate * awe

            # 执行LSTM步骤
            lstm_input = torch.cat([inputs.squeeze(1), awe], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))

            # 预测下一个词ID
            preds = self.fc(h)
            predicted = preds.argmax(dim=1)

            # 添加到结果列表
            sampled_ids.append(predicted.unsqueeze(1))

            # 更新下一轮输入
            inputs = self.embedding(predicted)

            # 检查是否所有batch都达到<end> (ID=2)
            if torch.all(torch.eq(predicted, 2)):
                break

        # 拼接所有预测ID
        sampled_ids_tensor = torch.cat(sampled_ids, dim=1)
        return sampled_ids_tensor

def main(args):
    # 确定计算设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载词汇表pickle
    with open(args.vocab_path, 'rb') as f:
        vocab: Vocabulary = pickle.load(f)
    print('词汇表加载完成')

    # 转换为JSON格式导出
    vocab_dict = {
        'idx2word': {int(k): v for k, v in vocab.idx2word.items()},
        'word2idx': {v: int(k) for v, k in vocab.word2idx.items()}
    }
    with open(args.output_vocab, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=4)
    print(f'词汇表已导出到: {args.output_vocab}')

    # 实例化模型（参数匹配训练时）
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_size,
        attention_dim=args.attention_dim,
        decoder_dim=args.decoder_dim,
        dropout=args.dropout
    )

    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print('模型加载完成')

    # 创建并转换为TorchScript
    inference_model = InferenceModule(model)
    scripted_model = torch.jit.script(inference_model)

    # 保存转换后的模型
    scripted_model.save(args.output_model)
    print(f'TorchScript模型已保存到: {args.output_model}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将PyTorch模型转换为TorchScript并导出词汇表')

    parser.add_argument('--model_path', type=str, default='models/final_model.pth.tar',
                        help='训练好的模型路径')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='词汇表路径')
    parser.add_argument('--output_model', type=str, default='./assets/model.ts',
                        help='输出TorchScript模型路径')
    parser.add_argument('--output_vocab', type=str, default='./assets/vocab.json',
                        help='输出词汇表JSON路径')

    # 模型维度参数
    parser.add_argument('--embed_size', type=int, default=512,
                        help='词嵌入维度')
    parser.add_argument('--attention_dim', type=int, default=512,
                        help='注意力维度')
    parser.add_argument('--decoder_dim', type=int, default=512,
                        help='解码器维度')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout率')

    args = parser.parse_args()
    main(args)
