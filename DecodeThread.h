#ifndef DECODETHREAD_H
#define DECODETHREAD_H

#include <QThread>
#include <QDebug>
#include <QDateTime>
#include <atomic> // 用于 std::atomic
#include <chrono> // 用于 std::chrono

// CUDA 运行时头文件
#include <cuda_runtime.h>

// FFmpeg 头文件 (C 语言兼容)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
}

class DecodeThread : public QThread
{
    Q_OBJECT
public:
    explicit DecodeThread(QObject *parent = nullptr);
    ~DecodeThread();

    // 设置 RTSP 地址
    void setUrl(const QString &url);
    // 停止线程
    void stop();

protected:
    // 线程入口
    void run() override;

signals:

    // 发送给 VideoOpenGLWidget 进行渲染
    void newFrameDisplayGPU(uint8_t* y, uint8_t* uv, int w, int h, int stride);

private:
    // ---------------------------------------------------------
    // 核心逻辑与辅助函数
    // ---------------------------------------------------------

    // GPU 解码主循环 (包含 ffmpeg 解码 -> CUDA 2D 拷贝逻辑)
    void runGpu();

    // 日志辅助
    void ffDebug(const QString &msg);

    // FFmpeg 回调：协商硬件加速格式 (自动选择 CUDA)
    static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);

    // FFmpeg 回调：处理读流超时 (防止网络卡死)
    static int ReadTimeoutCallback(void *ctx);

private:
    // ---------------------------------------------------------
    // 成员变量
    // ---------------------------------------------------------

    // 线程控制
    std::atomic<bool> stopFlag;     // 停止标志 (原子操作，线程安全)
    bool debugEnable;               // 调试开关
    QString url;                    // 视频流地址

    // 超时计算
    std::chrono::high_resolution_clock::time_point lastReadTime;

private:
    // --- 显存管理修改 ---
    // 两个缓冲区，交替读写，避免 cuda搬运时候解码器去写
    uint8_t* m_pBuffers[2] = {nullptr, nullptr};
    size_t m_bufferSize = 0; // 当前申请的缓冲区大小
    int m_bufIdx = 0;        // 当前正在使用的缓冲区索引 (0 或 1)

    // 记录当前的显存尺寸，用于检测分辨率变化
    int m_currentW = 0;
    int m_currentH = 0;
};

#endif // DECODETHREAD_H
