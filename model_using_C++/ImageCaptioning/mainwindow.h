#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#define slots Q_SLOTS
#include <QMainWindow>
#include <map>
#include <string>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onUploadClicked();
    void onGenerateClicked();

private:
    torch::Tensor loadImage(const QString& path);

    Ui::MainWindow *ui;
    QLabel *imageLabel;
    QPushButton *uploadButton;
    QPushButton *generateButton;
    QTextEdit *captionText;
    QString currentImagePath;
    torch::jit::Module module;
    std::map<int, std::string> idx2word;
    torch::Device device;
    bool modelLoaded;
    bool vocabLoaded;
};
#endif // MAINWINDOW_H


