#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include "DecodeThread.h"
#include "VideoOpenGLWidget.h"

// 最大支持的视频路数 (根据 UI 布局设定)
#define MAX_CNT 3

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

    // 系统初始化函数：绑定 UI、启动线程、连接信号
    int init();

private:
    Ui::Widget *ui;

    // --- 核心配置 ---
    QString            rtspUrl[MAX_CNT];       // 6路 RTSP 地址存储

    // --- 核心模块 (一一对应) ---
    DecodeThread* decodeThreads[MAX_CNT]; // 解码线程: 负责 拉流 -> NV12解码 -> GPU传输
    VideoOpenGLWidget* glWidgets[MAX_CNT];     // 显示控件: 负责 接收显存数据 -> OpenGL 渲染 (零拷贝)
};

#endif // WIDGET_H
