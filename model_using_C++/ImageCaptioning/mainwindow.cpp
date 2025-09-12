#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>
#include <vector>
#include <string>
#include <QImage>
#include <QApplication>
#include <QStyleFactory>
#include <QCoreApplication>

// 主窗口的构造函数，设置UI和加载模型
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , device(torch::kCPU)
    , modelLoaded(false)
    , vocabLoaded(false)
{
    ui->setupUi(this);  // 初始化UI从designer生成的

    // 设置整体现代风格，使用Fusion以获得平滑外观
    QApplication::setStyle(QStyleFactory::create("Fusion"));

    // 加载TorchScript模型从资源
    try {
        QString appDir = QCoreApplication::applicationDirPath();
        QString modelPath = appDir + "/assets/model.ts";
        module = torch::jit::load(modelPath.toStdString(), device);
        modelLoaded = true;  // 成功了就标记一下
    } catch (const std::exception& e) {
        modelLoaded = false;  // 出错了标记失败
    }

    // 加载词汇表从JSON资源
    QString appDir = QCoreApplication::applicationDirPath();
    QString vocabPath = appDir + "/assets/vocab.json";
    QFile vocabFile(vocabPath);
    if (vocabFile.open(QIODevice::ReadOnly)) {
        QJsonDocument doc = QJsonDocument::fromJson(vocabFile.readAll());
        QJsonObject obj = doc.object()["idx2word"].toObject();
        for (const QString& key : obj.keys()) {
            int idx = key.toInt();
            idx2word[idx] = obj[key].toString().toStdString();
        }
        vocabLoaded = true;  // 词汇加载好了
    } else {
        vocabLoaded = false;  // 词汇加载失败
    }

    // 设置中央widget和布局
    QWidget *central = new QWidget(this);
    setCentralWidget(central);
    QVBoxLayout *layout = new QVBoxLayout(central);
    layout->setContentsMargins(20, 20, 20, 20);  // 添加边距，让界面更宽敞现代
    layout->setSpacing(15);  // 增加组件间距

    // 图像显示标签，美化边框和阴影
    imageLabel = new QLabel();
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setStyleSheet("QLabel { background-color: white; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }");
    layout->addWidget(imageLabel);

    // 按钮布局，美化按钮为现代圆角风格
    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLayout->setSpacing(10);  // 按钮间距
    uploadButton = new QPushButton("上传图片");
    uploadButton->setStyleSheet("QPushButton { background-color: #007BFF; color: white; border-radius: 5px; padding: 8px 16px; font-size: 14px; } QPushButton:hover { background-color: #0056b3; }");
    btnLayout->addWidget(uploadButton);
    generateButton = new QPushButton("生成描述");
    generateButton->setStyleSheet("QPushButton { background-color: #28A745; color: white; border-radius: 5px; padding: 8px 16px; font-size: 14px; } QPushButton:hover { background-color: #218838; }");
    btnLayout->addWidget(generateButton);
    layout->addLayout(btnLayout);

    // 描述文本框，美化为现代卡片风格
    captionText = new QTextEdit();
    captionText->setReadOnly(true);
    captionText->setStyleSheet("QTextEdit { background-color: #f8f9fa; border: 1px solid #ddd; border-radius: 5px; padding: 10px; font-family: 'Segoe UI'; font-size: 14px; }");
    layout->addWidget(captionText);

    // 连接按钮信号
    connect(uploadButton, &QPushButton::clicked, this, &MainWindow::onUploadClicked);
    connect(generateButton, &QPushButton::clicked, this, &MainWindow::onGenerateClicked);

    // 设置窗口标题和最小尺寸，让整体更现代
    setWindowTitle("图像描述");
    setMinimumSize(400, 600);

    // 添加窗口图标，使用资源中的png文件
    setWindowIcon(QIcon(":/icons/ImageCaptioning.png"));
}

// 析构函数，清理UI
MainWindow::~MainWindow()
{
    delete ui;  // 释放UI资源
}

// 上传按钮点击，打开文件对话框选择图片
void MainWindow::onUploadClicked()
{
    currentImagePath = QFileDialog::getOpenFileName(this, "选择图片", "", "图像文件 (*.jpg *.png *.bmp)");
    if (!currentImagePath.isEmpty()) {
        QPixmap pixmap(currentImagePath);
        imageLabel->setPixmap(pixmap.scaled(300, 300, Qt::KeepAspectRatio));  // 显示缩放后的图片
        captionText->clear();  // 清空描述
    }
}

// 生成描述按钮点击
void MainWindow::onGenerateClicked()
{
    if (currentImagePath.isEmpty()) {
        captionText->setText("<font color='red'>请先上传图片</font>");  // 红色提示没图片
        return;
    }

    if (!modelLoaded || !vocabLoaded) {
        captionText->setText("<font color='red'>模型或词汇表加载失败</font>");  // 红色提示加载问题
        return;
    }

    auto tensor = loadImage(currentImagePath);  // 加载并预处理图像
    if (!tensor.defined()) {
        captionText->setText("<font color='red'>图片加载失败</font>");  // 红色提示加载失败
        return;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    inputs.push_back(20);  // 最大长度20

    try {
        auto output = module.forward(inputs).toTensor();
        auto ids = output.squeeze(0).to(torch::kCPU).to(torch::kInt64);

        std::vector<std::string> words;
        for (int i = 0; i < ids.size(0); ++i) {
            int id = ids[i].item<int64_t>();
            if (id == 1) continue;  // 忽略<start>
            if (id == 2) break;     // 遇到<end>停
            auto it = idx2word.find(id);
            if (it != idx2word.end()) {
                words.push_back(it->second);
            } else {
                words.push_back("<unk>");
            }
        }

        std::string caption;
        for (size_t j = 0; j < words.size(); ++j) {
            caption += words[j];
            if (j < words.size() - 1) caption += " ";
        }

        captionText->setText("<font color='green'>" + QString::fromStdString(caption) + "</font>");  // 绿色显示生成的描述
    } catch (const std::exception& e) {
        captionText->setText("<font color='red'>生成失败: " + QString(e.what()) + "</font>");  // 红色显示错误
    }
}

// 加载图像并转换为tensor，与Python预处理一致，使用QImage模拟PIL
torch::Tensor MainWindow::loadImage(const QString& path)
{
    QImage qimg(path);  // 用Qt加载图像，支持Unicode路径
    if (qimg.isNull()) {
        return torch::Tensor();  // 加载失败返回空
    }

    qimg = qimg.convertToFormat(QImage::Format_RGB888);  // 转RGB格式

    // 强制缩放到精确224x224，忽略宽高比，匹配Python Resize行为
    qimg = qimg.scaled(224, 224, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    // 手动转换为浮点[0,1]，创建tensor
    torch::Tensor tensor = torch::empty({1, 3, 224, 224}, torch::kFloat32);
    uchar* data = qimg.bits();  // 获取图像数据指针
    int bytesPerLine = qimg.bytesPerLine();  // 每行字节数

    for (int y = 0; y < 224; ++y) {  // 逐行复制到tensor
        uchar* row = data + y * bytesPerLine;  // 当前行指针
        for (int x = 0; x < 224; ++x) {
            tensor[0][0][y][x] = row[x*3 + 0] / 255.0f;  // R通道
            tensor[0][1][y][x] = row[x*3 + 1] / 255.0f;  // G通道
            tensor[0][2][y][x] = row[x*3 + 2] / 255.0f;  // B通道
        }
    }

    tensor = tensor.to(device);  // 移到设备

    auto mean = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, 3, 1, 1}).to(device);
    auto std = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, 3, 1, 1}).to(device);
    tensor.sub_(mean).div_(std);  // 归一化

    return tensor;
}
