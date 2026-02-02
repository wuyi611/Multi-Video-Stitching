#include "Widget.h"
#include "ui_Widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    init();
}

Widget::~Widget()
{
    delete ui;
}

int Widget::init()
{
    // 1. 绑定 UI 控件 (VideoOpenGLWidget)
    // 确保你的 ui 界面里，控件类型已经提升为 VideoOpenGLWidget
    glWidgets[0] = ui->openGLWidget_1;
    glWidgets[1] = ui->openGLWidget_2;
    glWidgets[2] = ui->openGLWidget_3;

    // 2. 设置 RTSP 地址 (保留主码流 stream0)
    rtspUrl[0] = "rtsp://192.168.9.81:554/11";
    rtspUrl[1] = "rtsp://192.168.9.82:554/11";
    rtspUrl[2] = "rtsp://192.168.9.83:554/11";

    pointPt[0] = QApplication::applicationDirPath() + "/calib_points_1.txt";
    pointPt[1] = QApplication::applicationDirPath() + "/calib_points_2.txt";
    pointPt[2] = QApplication::applicationDirPath() + "/calib_points_3.txt";

    for (int i = 0; i < 2; i++) {

        // 4. 初始化 解码线程
        decodeThreads[i] = new DecodeThread(this);
        decodeThreads[i]->setUrl(rtspUrl[i]);
        QString xmlPt = QApplication::applicationDirPath() + "/camera_params.xml";
        decodeThreads[i]->setCalibrationXmlPath(xmlPt);
        decodeThreads[i]->setEnableUndistort(true); // 开启/关闭矫正

        decodeThreads[i]->setIPMConfig(pointPt[i], glWidgets[i]->width(), glWidgets[i]->height());
        decodeThreads[i]->setEnableIPM(true); // 开启/关闭 IPM

        //5. 信号连接 (核心链路)

        //A. 解码 -> 显示 (GPU 零拷贝)
        connect(decodeThreads[i], &DecodeThread::newFrameDisplayGPU,
                glWidgets[i], &VideoOpenGLWidget::updateImageGPU);

        // 启动解码
        decodeThreads[i]->start();
    }

    return 0;
}
