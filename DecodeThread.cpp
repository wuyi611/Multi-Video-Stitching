#include "DecodeThread.h"

DecodeThread::DecodeThread(QObject *parent) : QThread(parent)
{
    stopFlag = false;
    debugEnable = true;
    url = "";
    m_xmlPath = "";
    m_isMapInit = false;
}

DecodeThread::~DecodeThread()
{
    stop();
    wait();

    // 释放 CUDA 资源
    cudaSetDevice(0);
    if (m_pBuffers[0]) { cudaFree(m_pBuffers[0]); m_pBuffers[0] = nullptr; }
    if (m_pBuffers[1]) { cudaFree(m_pBuffers[1]); m_pBuffers[1] = nullptr; }
    qDebug() << "[DecodeThread] 析构完毕，显存已释放";
}

void DecodeThread::setUrl(const QString &url) { this->url = url; }
void DecodeThread::setCalibrationXmlPath(const QString &path) { m_xmlPath = path; }
void DecodeThread::stop() { stopFlag = true; }

void DecodeThread::run() {
    avformat_network_init();
    runGpu();
}

void DecodeThread::ffDebug(const QString &msg) {
    if (debugEnable) qDebug() << "[DecodeThread]" << msg;
}

// 使用 Qt 解析 XML
CalibrationParams DecodeThread::loadParamsFromXml(const QString &path)
{
    CalibrationParams params;
    params.isValid = false;

    if (path.isEmpty()) return params;

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        ffDebug("无法打开 XML 文件: " + path);
        return params;
    }

    QDomDocument doc;
    QString errorStr;
    int errorLine, errorColumn;

    if (!doc.setContent(&file, false, &errorStr, &errorLine, &errorColumn)) {
        ffDebug(QString("XML 解析失败: %1 (Line %2)").arg(errorStr).arg(errorLine));
        file.close();
        return params;
    }
    file.close();

    QDomElement root = doc.documentElement();
    if (root.tagName() != "root") {
        ffDebug("XML 格式错误: 根节点不是 <root>");
        return params;
    }

    // 1. 解析 camera_matrix (3x3)
    QDomNode camNode = root.firstChildElement("camera_matrix");
    if (!camNode.isNull()) {
        params.K = cv::Mat::zeros(3, 3, CV_64F);
        QDomElement camElem = camNode.toElement();
        for (int i = 0; i < 9; i++) {
            QString tagName = QString("data%1").arg(i);
            QDomElement dataElem = camElem.firstChildElement(tagName);
            if (!dataElem.isNull()) {
                params.K.at<double>(i / 3, i % 3) = dataElem.text().toDouble();
            }
        }
    }

    // 2. [恢复] 解析 new_camera_matrix (3x3)
    QDomNode newCamNode = root.firstChildElement("new_camera_matrix");
    if (!newCamNode.isNull()) {
        params.NewK = cv::Mat::zeros(3, 3, CV_64F);
        QDomElement newCamElem = newCamNode.toElement();
        for (int i = 0; i < 9; i++) {
            QString tagName = QString("data%1").arg(i);
            QDomElement dataElem = newCamElem.firstChildElement(tagName);
            if (!dataElem.isNull()) {
                params.NewK.at<double>(i / 3, i % 3) = dataElem.text().toDouble();
            }
        }
    }

    // 3. 解析 camera_distortion (1x5)
    QDomNode distNode = root.firstChildElement("camera_distortion");
    if (!distNode.isNull()) {
        params.D = cv::Mat::zeros(1, 5, CV_64F);
        QDomElement distElem = distNode.toElement();
        for (int i = 0; i < 5; i++) {
            QString tagName = QString("data%1").arg(i);
            QDomElement dataElem = distElem.firstChildElement(tagName);
            if (!dataElem.isNull()) {
                params.D.at<double>(0, i) = dataElem.text().toDouble();
            }
        }
    }

    // 校验：K 和 NewK 都不为空才算有效
    if (!params.K.empty() && !params.NewK.empty() && cv::countNonZero(params.K) > 0) {
        params.isValid = true;
        ffDebug("XML 参数解析成功 (含 NewCameraMatrix)!");
    } else {
        ffDebug("XML 解析数据无效或缺少 NewCameraMatrix");
    }

    return params;
}

enum AVPixelFormat DecodeThread::get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts)
{
    Q_UNUSED(ctx);
    for (int i = 0; pix_fmts[i] != AV_PIX_FMT_NONE; i++) {
        if (pix_fmts[i] == AV_PIX_FMT_CUDA) return AV_PIX_FMT_CUDA;
    }
    return AV_PIX_FMT_NONE;
}

int DecodeThread::ReadTimeoutCallback(void *ctx)
{
    DecodeThread *self = static_cast<DecodeThread*>(ctx);
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - self->lastReadTime).count();
    if (elapsed > 3000 || self->stopFlag) return 1;
    return 0;
}

// =========================================================================
// [模块 1] FFmpeg 连接初始化
// =========================================================================
bool DecodeThread::initVideoConnection(FFmpegContext &ctx)
{
    if (url.isEmpty()) return false;

    AVDictionary *opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "probesize", "1024000", 0);

    ctx.fmtCtx = avformat_alloc_context();
    ctx.fmtCtx->interrupt_callback.callback = ReadTimeoutCallback;
    ctx.fmtCtx->interrupt_callback.opaque = this;
    lastReadTime = std::chrono::high_resolution_clock::now();

    if (avformat_open_input(&ctx.fmtCtx, url.toStdString().c_str(), nullptr, &opts) < 0) {
        av_dict_free(&opts);
        ffDebug("Open RTSP failed: " + url);
        return false;
    }
    av_dict_free(&opts);

    if (avformat_find_stream_info(ctx.fmtCtx, nullptr) < 0) return false;

    ctx.videoStreamIndex = av_find_best_stream(ctx.fmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ctx.videoStreamIndex < 0) return false;

    AVCodecParameters *codecPar = ctx.fmtCtx->streams[ctx.videoStreamIndex]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(codecPar->codec_id);
    if (!codec) return false;

    ctx.codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(ctx.codecCtx, codecPar);

    if (av_hwdevice_ctx_create(&ctx.hwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        ffDebug("CUDA HW init failed");
        return false;
    }
    ctx.codecCtx->hw_device_ctx = av_buffer_ref(ctx.hwDeviceCtx);
    ctx.codecCtx->get_format = get_hw_format;

    if (avcodec_open2(ctx.codecCtx, codec, nullptr) < 0) return false;

    return true;
}

// =========================================================================
// [模块 2] FFmpeg 资源释放
// =========================================================================
void DecodeThread::releaseVideoResources(FFmpegContext &ctx)
{
    if (ctx.codecCtx) {
        avcodec_free_context(&ctx.codecCtx);
        ctx.codecCtx = nullptr;
    }
    if (ctx.fmtCtx) {
        avformat_close_input(&ctx.fmtCtx);
        ctx.fmtCtx = nullptr;
    }
    if (ctx.hwDeviceCtx) {
        av_buffer_unref(&ctx.hwDeviceCtx);
        ctx.hwDeviceCtx = nullptr;
    }
}

// =========================================================================
// [模块 3] 分辨率检查与显存/映射表初始化
// =========================================================================
bool DecodeThread::checkResolutionAndInitResources(int width, int height, int stride)
{
    // 如果分辨率没变且缓冲区已分配，直接返回
    if (m_pBuffers[0] != nullptr && width == m_currentW && height == m_currentH) {
        return true;
    }

    ffDebug(QString("Resolution changed/Init: %1x%2").arg(width).arg(height));

    // 1. 重新分配 CUDA 双缓冲
    if (m_pBuffers[0]) cudaFree(m_pBuffers[0]);
    if (m_pBuffers[1]) cudaFree(m_pBuffers[1]);

    size_t y_size = (size_t)stride * height;
    size_t uv_size = (size_t)stride * height / 2;
    size_t total_size = y_size + uv_size;

    cudaMalloc((void**)&m_pBuffers[0], total_size);
    cudaMalloc((void**)&m_pBuffers[1], total_size);
    cudaMemset(m_pBuffers[0], 0, total_size);
    cudaMemset(m_pBuffers[1], 0, total_size);

    m_currentW = width;
    m_currentH = height;
    m_bufferSize = total_size;
    m_bufIdx = 0;

    // 2. 初始化畸变矫正映射表
    m_isMapInit = false; // 先置无效，直到加载成功
    if (!m_xmlPath.isEmpty()) {
        m_calibParams = loadParamsFromXml(m_xmlPath);

        if (m_calibParams.isValid) {
            ffDebug("正在根据 XML 生成 GPU 映射表...");
            cv::Mat mapX, mapY;

            // A. Y 平面映射表 (全分辨率)
            // 关键：直接使用 XML 里的 NewK，不自己计算
            cv::initUndistortRectifyMap(
                m_calibParams.K, m_calibParams.D, cv::Mat(),
                m_calibParams.NewK,
                cv::Size(width, height), CV_32FC1, mapX, mapY
                );

            // B. UV 平面映射表 (通过缩放 Y Map 获得，防止重影)
            // 这种方案能保证 UV 和 Y 的几何对齐最完美
            cv::Mat mapX_UV, mapY_UV;
            cv::resize(mapX, mapX_UV, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_LINEAR);
            cv::resize(mapY, mapY_UV, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_LINEAR);
            // 数值也要缩放
            mapX_UV = mapX_UV / 2.0f;
            mapY_UV = mapY_UV / 2.0f;

            // 上传到 GPU
            m_cudaMapX.upload(mapX);
            m_cudaMapY.upload(mapY);
            m_cudaMapX_UV.upload(mapX_UV);
            m_cudaMapY_UV.upload(mapY_UV);

            m_isMapInit = true;
            ffDebug("GPU 映射表初始化完成 (Load XML + Resize UV)");
        } else {
            ffDebug("XML 参数无效，将不执行矫正");
        }
    }

    // 3. 预分配 UV 通道分离所需的临时显存
    // 此处仅确保尺寸匹配时重新分配
    if (m_gpu_u_dst.size() != cv::Size(width/2, height/2)) {
        m_gpu_u_dst.create(height/2, width/2, CV_8UC1);
        m_gpu_v_dst.create(height/2, width/2, CV_8UC1);
    }

    return true;
}

// =========================================================================
// [模块 4] 执行 CUDA 矫正逻辑 (核心算法)
// =========================================================================
void DecodeThread::executeCudaCorrection(AVFrame *frame, uint8_t *dst_y, uint8_t *dst_uv, int width, int height, int stride)
{
    // 如果没有初始化映射表，执行简单的内存拷贝
    if (!m_isMapInit || m_cudaMapX.empty()) {
        cudaMemcpy2D(dst_y, stride, frame->data[0], stride, width, height, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(dst_uv, stride, frame->data[1], stride, width, height / 2, cudaMemcpyDeviceToDevice);
        return;
    }

    // --- 1. Y 平面矫正 (单通道) ---
    cv::cuda::GpuMat srcYMat(height, width, CV_8UC1, frame->data[0], stride);
    cv::cuda::GpuMat dstYMat(height, width, CV_8UC1, dst_y, stride);
    // Y 直接 Remap
    cv::cuda::remap(srcYMat, dstYMat, m_cudaMapX, m_cudaMapY, cv::INTER_LINEAR);

    // --- 2. UV 平面矫正 (拆分 -> Remap -> 合并) ---
    // 为了避开 OpenCV 不支持 CV_8UC2 Remap 的限制，并保证插值质量
    cv::cuda::GpuMat srcUVMat(height / 2, width / 2, CV_8UC2, frame->data[1], stride);
    cv::cuda::GpuMat dstUVMat(height / 2, width / 2, CV_8UC2, dst_uv, stride);

    // A. 拆分 (Split)
    std::vector<cv::cuda::GpuMat> src_split_vec;
    cv::cuda::split(srcUVMat, src_split_vec);
    // src_split_vec[0] 是 U, [1] 是 V

    // B. 分别 Remap (使用缩放后的 UV Map)
    // 注意：m_gpu_u_dst 和 m_gpu_v_dst 已经在 initResources 中分配好，这里直接使用
    cv::cuda::remap(src_split_vec[0], m_gpu_u_dst, m_cudaMapX_UV, m_cudaMapY_UV, cv::INTER_LINEAR);
    cv::cuda::remap(src_split_vec[1], m_gpu_v_dst, m_cudaMapX_UV, m_cudaMapY_UV, cv::INTER_LINEAR);

    // C. 合并 (Merge)
    std::vector<cv::cuda::GpuMat> dst_split_vec = {m_gpu_u_dst, m_gpu_v_dst};
    cv::cuda::merge(dst_split_vec, dstUVMat);
}

// =========================================================================
// [重构] 核心主循环：只负责调度
// =========================================================================
void DecodeThread::runGpu()
{
    cudaSetDevice(0);

    while (!stopFlag) {
        FFmpegContext ctx;
        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();

        // 1. 尝试建立连接
        if (!initVideoConnection(ctx)) {
            // 连接失败清理
            releaseVideoResources(ctx);
            av_packet_free(&pkt);
            av_frame_free(&frame);

            if (!stopFlag) {
                ffDebug("连接失败或中断，2秒后重试...");
                QThread::sleep(2);
            }
            continue; // 重试循环
        }

        ffDebug("初始化成功，开始解码循环...");

        // 2. 解码循环
        while (!stopFlag) {
            lastReadTime = std::chrono::high_resolution_clock::now();
            int ret = av_read_frame(ctx.fmtCtx, pkt);

            if (ret < 0) {
                if (ret == AVERROR(EAGAIN)) continue;
                ffDebug("读取流结束或出错");
                break; // 跳出内部循环，触发重连
            }

            if (pkt->stream_index == ctx.videoStreamIndex) {
                if (avcodec_send_packet(ctx.codecCtx, pkt) == 0) {
                    while (avcodec_receive_frame(ctx.codecCtx, frame) == 0) {
                        if (frame->format == AV_PIX_FMT_CUDA) {
                            int width = frame->width;
                            int height = frame->height;
                            int stride = frame->linesize[0];

                            // A. 检查分辨率 & 初始化显存/Map (如果是第一帧或分辨率变化)
                            checkResolutionAndInitResources(width, height, stride);

                            // B. 准备目标缓冲区
                            m_bufIdx = 1 - m_bufIdx; // 切换双缓冲
                            uint8_t* currentBuf = m_pBuffers[m_bufIdx];
                            uint8_t* dst_y = currentBuf;
                            uint8_t* dst_uv = currentBuf + ((size_t)stride * height);

                            // C. 执行矫正 (或拷贝)
                            executeCudaCorrection(frame, dst_y, dst_uv, width, height, stride);

                            // D. 发送信号
                            emit newFrameDisplayGPU(currentBuf, dst_uv, width, height, stride);
                        }
                    }
                }
            }
            av_packet_unref(pkt);
        }

        // 3. 资源清理
        releaseVideoResources(ctx);
        av_frame_free(&frame);
        av_packet_free(&pkt);
    }
}
