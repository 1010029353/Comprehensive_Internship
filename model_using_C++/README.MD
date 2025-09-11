# C++图像描述模型部署项目

@author SaltyHeart

这是一个使用C++和LibTorch实现的图像描述生成应用，加载Python训练的TorchScript模型进行推理。

## 项目目标
- 在C++环境中部署图像描述模型
- 提供图形界面（基于Qt）进行图像上传和描述生成

## 环境要求
- Qt 5.15+ (包括Qt Creator)
- LibTorch 1.7+ (CPU或GPU版本)
- C++17标准

## 安装与配置
1. **安装LibTorch**  
   从PyTorch官网下载LibTorch，解压到合适目录（如C:/libtorch）。  
   在Qt项目文件中配置路径（见ImageCaptioning.pro）。

2. **复制模型文件**  
   从Python项目复制 `assets/model.ts` 和 `assets/vocab.json` 到本项目的build目录。  
   （这些文件通过Python的convert.py生成）

## 运行流程
1. **打开项目**  
   使用Qt Creator打开 `ImageCaptioning.pro` 文件。

2. **构建项目**  
   配置构建套件（必须为MSVC），点击构建（Ctrl+B）。  
   确保LibTorch和OpenCV路径正确配置。

3. **运行应用**  
   点击运行（Ctrl+R），启动图形界面。  
   - 点击“选择图像”按钮上传图片。  
   - 应用会自动生成描述并显示。  
   - 支持拖拽图像文件到窗口。

4. **调试与日志**  
   - 检查控制台输出以查看推理过程。  
   - 如果模型加载失败，确认assets文件路径正确。

## 注意事项
- GPU版本需要CUDA支持，并配置LibTorch的CUDA版本。
- 图像输入需为RGB格式，尺寸会自动调整。
- 如需自定义模型路径，修改mainwindow.cpp中的加载代码。
- 项目日志和错误信息会输出到控制台。

更多细节见代码注释和mainwindow.cpp
