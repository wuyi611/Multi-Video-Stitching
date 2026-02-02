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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
}

// --- 参数结构体 ---

// 1. 畸变矫正参数 (内参)
struct CalibrationParams {
    cv::Mat K;           // 原始内参
    cv::Mat D;           // 畸变系数
    cv::Mat NewK;        // 矫正后的新内参
    bool isValid = false;
};

// 2. 逆透视变换参数 (IPM)
struct IPMParams {
    std::vector<cv::Point2f> srcPoints; // 源点 (来自 txt)
    cv::Size outSize;                   // 输出俯视图大小
    cv::Mat homography;                 // 3x3 变换矩阵 (Y通道)
    cv::Mat homographyUV;               // 3x3 变换矩阵 (UV通道，缩小版)
    bool isValid = false;
};

// FFmpeg 上下文
struct FFmpegContext {
    AVFormatContext *fmtCtx = nullptr;
    AVCodecContext *codecCtx = nullptr;
    AVBufferRef *hwDeviceCtx = nullptr;
    int videoStreamIndex = -1;
};

class DecodeThread : public QThread
{
    Q_OBJECT
public:
    explicit DecodeThread(QObject *parent = nullptr);
    ~DecodeThread();

    // --- 设置接口 ---
    void setUrl(const QString &url);

    // 设置畸变矫正 XML
    void setCalibrationXmlPath(const QString &path);

    // 设置 IPM 坐标文件 (txt) 和输出尺寸
    void setIPMConfig(const QString &txtPath, int outW, int outH);

    // 功能开关
    void setEnableUndistort(bool enable);
    void setEnableIPM(bool enable);

    void stop();

protected:
    void run() override;

signals:
    // 发出的信号是最终处理后的显存指针
    void newFrameDisplayGPU(uint8_t* y, uint8_t* uv, int w, int h, int stride);

private:
    void runGpu();
    void ffDebug(const QString &msg);
    static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts);
    static int ReadTimeoutCallback(void *ctx);

    // --- 资源初始化 ---
    bool initVideoConnection(FFmpegContext &ctx);
    void releaseVideoResources(FFmpegContext &ctx);

    // 统一资源检查入口
    bool checkResolutionAndInitResources(int width, int height, int stride);

    // 分模块初始化
    void initUndistortResources(int width, int height);
    void initIPMResources(int width, int height); // 需根据图像尺寸计算 UV 矩阵

    // --- 加载辅助 ---
    CalibrationParams loadParamsFromXml(const QString &path);
    std::vector<cv::Point2f> loadPointsFromTxt(const QString &path);

    // --- CUDA 核心算法 (分离) ---
    // 1. 执行畸变矫正: src -> dst
    void executeCudaUndistort(const cv::cuda::GpuMat &srcY, const cv::cuda::GpuMat &srcUV,
                              cv::cuda::GpuMat &dstY, cv::cuda::GpuMat &dstUV);

    // 2. 执行逆透视: src -> dst
    void executeCudaIPM(const cv::cuda::GpuMat &srcY, const cv::cuda::GpuMat &srcUV,
                        cv::cuda::GpuMat &dstY, cv::cuda::GpuMat &dstUV);

private:
    std::atomic<bool> stopFlag;
    bool debugEnable;
    QString url;
    QString m_xmlPath;
    QString m_ipmTxtPath;
    cv::Size m_ipmOutSize;

    // 功能开关
    bool m_enableUndistort = true;
    bool m_enableIPM = false;

    std::chrono::high_resolution_clock::time_point lastReadTime;

    // --- 显存管理 ---
    // 输出双缓冲 (用于显示)
    uint8_t* m_pBuffers[2] = {nullptr, nullptr};
    int m_bufIdx = 0;

    // 中间缓冲区 (用于 Undistort -> IPM 的串联)
    // 只有当两者都开启时才使用
    uint8_t* m_pMidBuffer = nullptr;

    int m_currentW = 0;
    int m_currentH = 0;
    size_t m_frameSize = 0;

    // --- 畸变矫正资源 ---
    CalibrationParams m_calibParams;
    bool m_isUndistortInit = false;
    cv::cuda::GpuMat m_cudaMapX, m_cudaMapY;       // Y Map
    cv::cuda::GpuMat m_cudaMapX_UV, m_cudaMapY_UV; // UV Map

    // --- IPM 资源 ---
    IPMParams m_ipmParams;
    bool m_isIPMInit = false;

    // --- 通用临时变量 (UV处理用) ---
    cv::cuda::GpuMat m_gpu_u_split, m_gpu_v_split;
    cv::cuda::GpuMat m_gpu_u_merge, m_gpu_v_merge;
};

#endif // DECODETHREAD_H
