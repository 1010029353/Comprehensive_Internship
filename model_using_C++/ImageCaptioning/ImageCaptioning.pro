QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

RESOURCES += \
    resource.qrc

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += .\libtorch\include
INCLUDEPATH += .\libtorch\include\torch\csrc\api\include

LIBS += -L"E:\Comprehensive_Internship\model_using_C++\ImageCaptioning\libtorch\lib" \
            -lc10\
            -lc10_cuda\
            -ltorch\
            -ltorch_cuda\
            -ltorch_cpu

LIBS += -INCLUDE:"?ignore_this_library_placeholder@@YAHXZ"

#程序版本
VERSION = 1.0.0
#程序图标
RC_ICONS = favicon.ico
#公司名称
QMAKE_TARGET_COMPANY = "Mars"
#程序说明
QMAKE_TARGET_DESCRIPTION = "ImageCaptioning using"
#版权信息
QMAKE_TARGET_COPYRIGHT = "Copyright(C) 2025"
#程序名称
QMAKE_TARGET_PRODUCT = "ImageCaptioning"
#程序语言
#0x0800代表和系统当前语言一致
RC_LANG = 0x0800
