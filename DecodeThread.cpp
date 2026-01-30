#include "DecodeThread.h"



DecodeThread::DecodeThread(QObject *parent) : QThread(parent)
{
    stopFlag = false;
    debugEnable = true; // 默认开启调试
    url = "";
}

DecodeThread::~DecodeThread()
{
    stop();
    wait(); // 等待 runGpu 退出

    // 显式切换上下文并释放资源
    cudaSetDevice(0);

    if (m_pBuffers[0]) {
        cudaFree(m_pBuffers[0]);
        m_pBuffers[0] = nullptr;
    }
    if (m_pBuffers[1]) {
        cudaFree(m_pBuffers[1]);
        m_pBuffers[1] = nullptr;
    }
    qDebug() << "[DecodeThread] 析构完毕，显存已释放";
}

void DecodeThread::setUrl(const QString &url)
{
    this->url = url;
}

void DecodeThread::stop()
{
    stopFlag = true;
}

void DecodeThread::run()
{
    // 初始化网络库 (全局只需一次，放在这里确保线程内可用)
    avformat_network_init();
    runGpu();
}

void DecodeThread::ffDebug(const QString &msg)
{
    if (debugEnable) {
        qDebug() << "[DecodeThread]" << msg;
    }
}

// 硬件加速回调：优先选择 CUDA
enum AVPixelFormat DecodeThread::get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    Q_UNUSED(ctx);
    for (int i = 0; pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
        if (pix_fmts[i] == AV_PIX_FMT_CUDA) {
            return AV_PIX_FMT_CUDA;
        }
    }
    return AV_PIX_FMT_NONE;
}

// 超时中断回调
int DecodeThread::ReadTimeoutCallback(void *ctx)
{
    DecodeThread *self = static_cast<DecodeThread*>(ctx);
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - self->lastReadTime).count();

    // 超过3秒无数据或收到停止信号，中断阻塞
    if (elapsed > 3000 || self->stopFlag) {
        return 1; // 返回 1 表示中断
    }
    return 0;
}

void DecodeThread::runGpu()
{
    // [核心] 必须先绑定 CUDA 设备
    cudaSetDevice(0);

    while (!stopFlag) {
        // 定义 FFmpeg 对象
        AVFormatContext *fmtCtx = nullptr;
        AVCodecContext *codecCtx = nullptr;
        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();
        AVBufferRef *hwDeviceCtx = nullptr;
        const AVCodec *codec = nullptr;
        int videoStream = -1;

        // -------------------------------------------------------------
        // 1. 打开流与解码器初始化
        // -------------------------------------------------------------
        do {
            if (url.isEmpty()) { QThread::msleep(100); break; }

            AVDictionary *opts = nullptr;
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            av_dict_set(&opts, "flags", "low_delay", 0);
            av_dict_set(&opts, "fflags", "nobuffer", 0);
            // 增大一点探测尺寸，防止复杂流探测失败
            av_dict_set(&opts, "probesize", "1024000", 0);

            fmtCtx = avformat_alloc_context();
            fmtCtx->interrupt_callback.callback = ReadTimeoutCallback;
            fmtCtx->interrupt_callback.opaque = this;
            lastReadTime = std::chrono::high_resolution_clock::now();

            // 打开输入
            if (avformat_open_input(&fmtCtx, url.toStdString().c_str(), nullptr, &opts) < 0) {
                av_dict_free(&opts);
                ffDebug("无法打开 RTSP 流: " + url);
                break;
            }
            av_dict_free(&opts);

            if (avformat_find_stream_info(fmtCtx, nullptr) < 0) break;

            videoStream = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            if (videoStream < 0) break;

            AVCodecParameters *codecPar = fmtCtx->streams[videoStream]->codecpar;
            codec = avcodec_find_decoder(codecPar->codec_id);
            if (!codec) break;

            codecCtx = avcodec_alloc_context3(codec);
            avcodec_parameters_to_context(codecCtx, codecPar);

            // [核心] 创建 CUDA 硬件上下文
            if (av_hwdevice_ctx_create(&hwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) >= 0) {
                codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx);
                codecCtx->get_format = get_hw_format;
            } else {
                ffDebug("创建 CUDA 硬件设备失败");
                break;
            }

            if (avcodec_open2(codecCtx, codec, nullptr) < 0) break;

            ffDebug("初始化成功，开始解码循环...");

            // -------------------------------------------------------------
            // 2. 解码循环 (av_read_frame)
            // -------------------------------------------------------------
            while (!stopFlag) {
                lastReadTime = std::chrono::high_resolution_clock::now();
                int ret = av_read_frame(fmtCtx, pkt);

                if (ret < 0) {
                    if (ret == AVERROR(EAGAIN)) continue;
                    ffDebug("读取帧失败或流结束");
                    break; // 跳出内层循环，触发重连
                }

                if (pkt->stream_index == videoStream) {
                    if (avcodec_send_packet(codecCtx, pkt) == 0) {
                        while (avcodec_receive_frame(codecCtx, frame) == 0) {

                            // 确认这是显存中的帧 (NV12)
                            if (frame->format == AV_PIX_FMT_CUDA) {
                                int width = frame->width;
                                int height = frame->height;
                                int srcStride = frame->linesize[0]; // GPU Pitch

                                // 为了最快拷贝，目标步长保持与源步长一致
                                int dstStride = srcStride;

                                // 计算所需总大小 (NV12 = Y + UV)
                                size_t y_size = (size_t)dstStride * height;
                                size_t uv_size = (size_t)dstStride * height / 2;
                                size_t total_size = y_size + uv_size;

                                // 仅在分辨率变化时分配显存
                                if (m_pBuffers[0] == nullptr || width != m_currentW || height != m_currentH) {
                                    ffDebug("分辨率变化/初次初始化，分配双缓冲...");

                                    // 安全释放旧的
                                    if (m_pBuffers[0]) cudaFree(m_pBuffers[0]);
                                    if (m_pBuffers[1]) cudaFree(m_pBuffers[1]);

                                    // 分配新的双缓冲
                                    cudaMalloc((void**)&m_pBuffers[0], total_size);
                                    cudaMalloc((void**)&m_pBuffers[1], total_size);

                                    // 显存清零 (防止黑屏/花屏)
                                    cudaMemset(m_pBuffers[0], 0, total_size);
                                    cudaMemset(m_pBuffers[1], 0, total_size);

                                    m_currentW = width;
                                    m_currentH = height;
                                    m_bufferSize = total_size;
                                    m_bufIdx = 0;
                                }

                                // 双缓冲切换 (Ping-Pong)
                                // 切换到“后台”缓冲区进行写入，不影响前台 OpenGL 读取
                                m_bufIdx = 1 - m_bufIdx;
                                uint8_t* currentBuf = m_pBuffers[m_bufIdx];

                                uint8_t* dst_y = currentBuf;
                                uint8_t* dst_uv = currentBuf + y_size;

                                // D2D 拷贝 (Deep Copy)
                                // 必须深拷贝，因为 av_frame_unref 会让 frame->data 失效
                                cudaMemcpy2D(dst_y, dstStride, frame->data[0], srcStride, width, height, cudaMemcpyDeviceToDevice);
                                cudaMemcpy2D(dst_uv, dstStride, frame->data[1], srcStride, width, height / 2, cudaMemcpyDeviceToDevice);

                                // 发送当前缓冲区的指针
                                // 此时 OpenGL 可能还在读另一个 buffer (1 - m_bufIdx)，绝对安全
                                emit newFrameDisplayGPU(currentBuf, dst_uv, width, height, dstStride);
                            }
                        }
                    }
                }
                av_packet_unref(pkt);
            } // end while read frame

        } while (false); // end do...while

        // -------------------------------------------------------------
        // 3. 资源清理 (网络断开时)
        // -------------------------------------------------------------
        // 注意：千万不要在这里 cudaFree m_pBuffers！
        // 否则 UI 线程刚好刷新上一帧时会访问野指针导致崩溃。

        if (frame) av_frame_free(&frame);
        if (pkt) av_packet_free(&pkt);
        if (codecCtx) avcodec_free_context(&codecCtx);
        if (fmtCtx) avformat_close_input(&fmtCtx);
        if (hwDeviceCtx) av_buffer_unref(&hwDeviceCtx);

        if (!stopFlag) {
            ffDebug("连接中断，2秒后重试...");
            QThread::sleep(2);
        }
    }
}
