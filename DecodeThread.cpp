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

void DecodeThread::runGpu()
{
    cudaSetDevice(0);

    while (!stopFlag) {
        AVFormatContext *fmtCtx = nullptr;
        AVCodecContext *codecCtx = nullptr;
        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();
        AVBufferRef *hwDeviceCtx = nullptr;
        const AVCodec *codec = nullptr;
        int videoStream = -1;

        do {
            if (url.isEmpty()) { QThread::msleep(100); break; }

            AVDictionary *opts = nullptr;
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            av_dict_set(&opts, "flags", "low_delay", 0);
            av_dict_set(&opts, "fflags", "nobuffer", 0);
            av_dict_set(&opts, "probesize", "1024000", 0);

            fmtCtx = avformat_alloc_context();
            fmtCtx->interrupt_callback.callback = ReadTimeoutCallback;
            fmtCtx->interrupt_callback.opaque = this;
            lastReadTime = std::chrono::high_resolution_clock::now();

            if (avformat_open_input(&fmtCtx, url.toStdString().c_str(), nullptr, &opts) < 0) {
                av_dict_free(&opts); ffDebug("Open RTSP failed: " + url); break;
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

            if (av_hwdevice_ctx_create(&hwDeviceCtx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) >= 0) {
                codecCtx->hw_device_ctx = av_buffer_ref(hwDeviceCtx);
                codecCtx->get_format = get_hw_format;
            } else { ffDebug("CUDA HW init failed"); break; }

            if (avcodec_open2(codecCtx, codec, nullptr) < 0) break;
            ffDebug("初始化成功，开始解码循环...");

            while (!stopFlag) {
                lastReadTime = std::chrono::high_resolution_clock::now();
                int ret = av_read_frame(fmtCtx, pkt);
                if (ret < 0) { if (ret == AVERROR(EAGAIN)) continue; ffDebug("Stream end/error"); break; }

                if (pkt->stream_index == videoStream) {
                    if (avcodec_send_packet(codecCtx, pkt) == 0) {
                        while (avcodec_receive_frame(codecCtx, frame) == 0) {
                            if (frame->format == AV_PIX_FMT_CUDA) {
                                int width = frame->width;
                                int height = frame->height;
                                int srcStride = frame->linesize[0];

                                // ------------------------------------------------
                                // 1. 显存初始化与映射表生成
                                // ------------------------------------------------
                                if (m_pBuffers[0] == nullptr || width != m_currentW || height != m_currentH) {
                                    ffDebug(QString("Resolution changed: %1x%2").arg(width).arg(height));

                                    // 释放旧显存
                                    if (m_pBuffers[0]) cudaFree(m_pBuffers[0]);
                                    if (m_pBuffers[1]) cudaFree(m_pBuffers[1]);

                                    int dstStride = srcStride;
                                    size_t y_size = (size_t)dstStride * height;
                                    size_t uv_size = (size_t)dstStride * height / 2;
                                    size_t total_size = y_size + uv_size;

                                    // 分配新双缓冲
                                    cudaMalloc((void**)&m_pBuffers[0], total_size);
                                    cudaMalloc((void**)&m_pBuffers[1], total_size);
                                    cudaMemset(m_pBuffers[0], 0, total_size);
                                    cudaMemset(m_pBuffers[1], 0, total_size);

                                    m_currentW = width;
                                    m_currentH = height;
                                    m_bufferSize = total_size;
                                    m_bufIdx = 0;

                                    // ------------------------------------------------
                                    // 2. [关键] 读取 XML 并生成 Map (直接使用 NewK)
                                    // ------------------------------------------------
                                    if (!m_xmlPath.isEmpty()) {
                                        m_calibParams = loadParamsFromXml(m_xmlPath);

                                        if (m_calibParams.isValid) {
                                            cv::Mat mapX, mapY;

                                            ffDebug("正在使用 XML 中的新内参生成映射表...");

                                            // A. 生成 Y 平面映射表 (全分辨率)
                                            // 直接使用 XML 里读出来的 m_calibParams.NewK
                                            cv::initUndistortRectifyMap(
                                                m_calibParams.K,
                                                m_calibParams.D,
                                                cv::Mat(),
                                                m_calibParams.NewK, // <--- 关键修改：直接用 XML 参数
                                                cv::Size(width, height),
                                                CV_32FC1,
                                                mapX,
                                                mapY
                                                );

                                            // B. 生成 UV 平面映射表 (Resize 方案 - 防重影)
                                            // 直接缩放 Y 的 Map，这是保证 Y 和 UV 严格对齐的最佳方法
                                            cv::Mat mapX_UV, mapY_UV;

                                            // 尺寸缩小一半
                                            cv::resize(mapX, mapX_UV, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_LINEAR);
                                            cv::resize(mapY, mapY_UV, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_LINEAR);

                                            // 数值缩小一半
                                            mapX_UV = mapX_UV / 2.0f;
                                            mapY_UV = mapY_UV / 2.0f;

                                            // 上传 GPU
                                            m_cudaMapX.upload(mapX);
                                            m_cudaMapY.upload(mapY);
                                            m_cudaMapX_UV.upload(mapX_UV);
                                            m_cudaMapY_UV.upload(mapY_UV);

                                            m_isMapInit = true;
                                            ffDebug("GPU 畸变矫正表初始化完成 (使用 XML 参数 + Resize UV)");
                                        } else {
                                            m_isMapInit = false;
                                            ffDebug("XML 参数无效或缺少 NewCameraMatrix，跳过矫正");
                                        }
                                    }
                                }

                                // ------------------------------------------------
                                // 3. 实时矫正与双缓冲写入
                                // ------------------------------------------------
                                m_bufIdx = 1 - m_bufIdx;
                                uint8_t* currentBuf = m_pBuffers[m_bufIdx];
                                uint8_t* dst_y = currentBuf;
                                uint8_t* dst_uv = currentBuf + ((size_t)srcStride * height);

                                if (m_isMapInit && !m_cudaMapX.empty()) {
                                    // --- Y 平面 (CV_8UC1) ---
                                    cv::cuda::GpuMat srcYMat(height, width, CV_8UC1, frame->data[0], srcStride);
                                    cv::cuda::GpuMat dstYMat(height, width, CV_8UC1, dst_y, srcStride);

                                    // Y 直接 Remap
                                    cv::cuda::remap(srcYMat, dstYMat, m_cudaMapX, m_cudaMapY, cv::INTER_LINEAR);

                                    // --- UV 平面 (CV_8UC2 -> 拆分 -> 矫正 -> 合并) ---
                                    cv::cuda::GpuMat srcUVMat(height / 2, width / 2, CV_8UC2, frame->data[1], srcStride);
                                    cv::cuda::GpuMat dstUVMat(height / 2, width / 2, CV_8UC2, dst_uv, srcStride);

                                    // A. 拆分通道 (Split)
                                    // 使用成员变量 m_gpu_u, m_gpu_v 避免重复 malloc
                                    std::vector<cv::cuda::GpuMat> src_split_vec;
                                    // split 需要 vector 作为输出容器，但我们可以让它指向预分配的 GpuMat
                                    // (OpenCV split 如果发现尺寸类型匹配，会重用内存)
                                    cv::cuda::split(srcUVMat, src_split_vec);
                                    // 注意：这里 src_split_vec 里的 GpuMat 是 split 内部临时生成的还是引用的，
                                    // 取决于实现。为了稳妥，我们可以直接让它们持有数据。

                                    // 确保目标单通道缓存已分配
                                    if (m_gpu_u_dst.size() != cv::Size(width/2, height/2)) {
                                        m_gpu_u_dst.create(height/2, width/2, CV_8UC1);
                                        m_gpu_v_dst.create(height/2, width/2, CV_8UC1);
                                    }

                                    // B. 分别矫正
                                    // src_split_vec[0] 是 U, [1] 是 V
                                    cv::cuda::remap(src_split_vec[0], m_gpu_u_dst, m_cudaMapX_UV, m_cudaMapY_UV, cv::INTER_LINEAR);
                                    cv::cuda::remap(src_split_vec[1], m_gpu_v_dst, m_cudaMapX_UV, m_cudaMapY_UV, cv::INTER_LINEAR);

                                    // C. 合并通道 (Merge)
                                    std::vector<cv::cuda::GpuMat> dst_split_vec = {m_gpu_u_dst, m_gpu_v_dst};
                                    cv::cuda::merge(dst_split_vec, dstUVMat);

                                } else {
                                    // [无标定] 原样拷贝
                                    cudaMemcpy2D(dst_y, srcStride, frame->data[0], srcStride, width, height, cudaMemcpyDeviceToDevice);
                                    cudaMemcpy2D(dst_uv, srcStride, frame->data[1], srcStride, width, height / 2, cudaMemcpyDeviceToDevice);
                                }

                                // ------------------------------------------------
                                // 4. 发送给 UI 渲染
                                // ------------------------------------------------
                                emit newFrameDisplayGPU(currentBuf, dst_uv, width, height, srcStride);
                            }
                        }
                    }
                }
                av_packet_unref(pkt);
            }
        } while (false);

        // 资源清理
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
