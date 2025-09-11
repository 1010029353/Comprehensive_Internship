# Python图像描述模型训练项目

@author ZhangZiYang

这是一个使用Python实现的图像描述生成模型训练项目，基于PyTorch框架，采用注意力机制的编码器-解码器架构。

## 项目目标
- 使用COCO数据集训练图像描述模型
- 支持模型转换为TorchScript格式，用于后续部署

## 环境要求
- Python 3.8+
- PyTorch 1.7+
- 其他依赖见requirements.txt

## 标准运行流程
1. **安装依赖**  
   运行 `pip install -r requirements.txt` 安装所需库。

2. **下载COCO数据集**  
   运行 `python image_captioning/download_coco_dataset.py` 下载2014训练/验证集和标注文件。  
   数据将保存在 `data/` 目录下。

3. **调整图像大小**  
   运行 `python image_captioning/resize.py --image_dir data/train2014/ --output_dir data/resized2014/`  
   和 `python image_captioning/resize.py --image_dir data/val2014/ --output_dir data/resizedval2014/`  
   将图像调整为模型输入尺寸（默认256x256）。

4. **构建词汇表**  
   运行 `python image_captioning/build_vocab.py --caption_path data/annotations/captions_train2014.json`  
   生成词汇表文件 `data/vocab.pkl`。

5. **训练模型**  
   运行 `python image_captioning/train.py --image_dir data/resized2014 --caption_path data/annotations/captions_train2014.json --val_image_dir data/resizedval2014 --val_caption_path data/annotations/captions_val2014.json`  
   开始训练。模型检查点保存在 `models/` 目录。

6. **生成描述**  
   运行 `python image_captioning/sample.py --image_path <图像路径> --model_path models/final_model.pth.tar`  
   为单张图像生成描述。

7. **模型转换（若无部署模型需求忽略此步骤）**  
   运行 `python image_captioning/convert.py --model_path models/final_model.pth.tar`  
   将模型转换为TorchScript格式，保存为 `assets/model.ts`，并导出词汇表JSON。

## 注意事项
- 训练前确保有足够的GPU资源。
- 如需自定义参数，请查看各脚本的argparse帮助。
- 项目日志保存在 `logs/` 目录，可用TensorBoard查看。

更多细节见代码注释。
