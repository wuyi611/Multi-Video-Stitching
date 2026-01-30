#ifndef DECODETHREAD_H
#define DECODETHREAD_H

#include <QThread>
#include <QDebug>
#include <QDateTime>
#include <QFile>
#include <atomic>
#include <chrono>

// Qt XML 模块
#include <QtXml>
#include <QDomDocument>

// OpenCV 核心、CUDA 模块
#include <opencv2/opencv.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

// CUDA 运行时
#include <cuda_runtime.h>

// FFmpeg (C 语言兼容)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
}

// 标定参数结构体
struct CalibrationParams {
    cv::Mat K;           // 原始内参
    cv::Mat D;           // 畸变系数
    cv::Rect roi;        // 感兴趣区域
    bool isValid = false;
};

class DecodeThread : public QThread
{
    Q_OBJECT
public:
    explicit DecodeThread(QObject *parent = nullptr);
    ~DecodeThread();

    // 设置 RTSP 地址
    void setUrl(const QString &url);
    // 设置标定 XML 文件路径
    void setCalibrationXmlPath(const QString &path);
    // 停止线程
    void stop();

protected:
    void run() override;

signals:
    // 发出的信号是矫正后的画面 (显存指针)
    void newFrameDisplayGPU(uint8_t* y, uint8_t* uv, int w, int h, int stride);

private:
    void runGpu();
    void ffDebug(const QString &msg);
    static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);
    static int ReadTimeoutCallback(void *ctx);

    // XML 解析辅助函数 (使用 QDomDocument)
    CalibrationParams loadParamsFromXml(const QString &path);

private:
    std::atomic<bool> stopFlag;
    bool debugEnable;
    QString url;
    QString m_xmlPath;

    std::chrono::high_resolution_clock::time_point lastReadTime;

    // --- 显存管理 (双缓冲) ---
    uint8_t* m_pBuffers[2] = {nullptr, nullptr};
    size_t m_bufferSize = 0;
    int m_bufIdx = 0;
    int m_currentW = 0;
    int m_currentH = 0;

    // --- 畸变矫正资源 ---
    CalibrationParams m_calibParams;     // 参数缓存
    bool m_isMapInit = false;            // 映射表是否初始化
    cv::cuda::GpuMat m_cudaMapX, m_cudaMapY; // GPU 映射表
    cv::cuda::GpuMat m_cudaMapX_UV, m_cudaMapY_UV; // UV 专用映射表
    cv::cuda::GpuMat m_gpu_u, m_gpu_v;      // 分离后的源 U, V
    cv::cuda::GpuMat m_gpu_u_dst, m_gpu_v_dst; // 矫正后的 U, V
};

#endif // DECODETHREAD_H
