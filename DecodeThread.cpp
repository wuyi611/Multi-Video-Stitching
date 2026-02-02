#include "DecodeThread.h"
#include <QTextStream>

DecodeThread::DecodeThread(QObject *parent) : QThread(parent)
{
    stopFlag = false;
    debugEnable = true;
    url = "";
    m_xmlPath = "";
    m_ipmTxtPath = "";
    m_ipmOutSize = cv::Size(600, 800); // 默认值
}

DecodeThread::~DecodeThread()
{
    stop();
    wait();

    cudaSetDevice(0);
    if (m_pBuffers[0]) cudaFree(m_pBuffers[0]);
    if (m_pBuffers[1]) cudaFree(m_pBuffers[1]);
    if (m_pMidBuffer)  cudaFree(m_pMidBuffer);
    qDebug() << "[DecodeThread] 显存已释放";
}

// --- 设置接口 ---
void DecodeThread::setUrl(const QString &url) { this->url = url; }
void DecodeThread::setCalibrationXmlPath(const QString &path) { m_xmlPath = path; }
void DecodeThread::setIPMConfig(const QString &txtPath, int outW, int outH) {
    m_ipmTxtPath = txtPath;
    m_ipmOutSize = cv::Size(outW, outH);
}
void DecodeThread::setEnableUndistort(bool enable) { m_enableUndistort = enable; }
void DecodeThread::setEnableIPM(bool enable) { m_enableIPM = enable; }
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
// [新增] 读取 TXT 坐标文件
// =========================================================================
std::vector<cv::Point2f> DecodeThread::loadPointsFromTxt(const QString &path) {
    std::vector<cv::Point2f> points;
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        ffDebug("无法打开 IPM TXT: " + path);
        return points;
    }

    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine();
        if (line == "None") continue;
        QStringList parts = line.split(",");
        if (parts.size() >= 2) {
            float x = parts[0].toFloat();
            float y = parts[1].toFloat();
            points.push_back(cv::Point2f(x, y));
        }
    }
    file.close();
    return points;
}

// =========================================================================
// [修改] 资源初始化入口 (分辨率变化时调用)
// =========================================================================
bool DecodeThread::checkResolutionAndInitResources(int width, int height, int stride)
{
    // 分辨率未变且已分配，直接返回
    if (m_pBuffers[0] != nullptr && width == m_currentW && height == m_currentH) {
        return true;
    }

    ffDebug(QString("初始化显存资源: %1x%2").arg(width).arg(height));

    // 1. 释放旧显存
    if (m_pBuffers[0]) cudaFree(m_pBuffers[0]);
    if (m_pBuffers[1]) cudaFree(m_pBuffers[1]);
    if (m_pMidBuffer)  cudaFree(m_pMidBuffer);

    // 2. 计算大小 (NV12)
    m_currentW = width;
    m_currentH = height;
    size_t y_size = (size_t)stride * height;
    size_t uv_size = (size_t)stride * height / 2;
    m_frameSize = y_size + uv_size;

    // 3. 分配显示用的双缓冲
    // 注意：如果开启了 IPM，输出尺寸可能会变。
    // 这里为了简单，我们让显存足够大，或者根据 IPM 输出动态调整。
    // 为了兼容性，我们分配 Max(原图大小, IPM输出大小)
    size_t allocSize = m_frameSize;
    if (m_enableIPM) {
        size_t ipm_size = (size_t)m_ipmOutSize.width * m_ipmOutSize.height * 3 / 2; // 估算
        if (ipm_size > allocSize) allocSize = ipm_size;
    }

    cudaMalloc((void**)&m_pBuffers[0], allocSize);
    cudaMalloc((void**)&m_pBuffers[1], allocSize);

    // 4. 分配中间缓冲 (仅当两者都开启时需要)
    if (m_enableUndistort && m_enableIPM) {
        // 中间缓冲存放 Undistort 的结果，所以是原图大小
        cudaMalloc((void**)&m_pMidBuffer, m_frameSize);
    } else {
        m_pMidBuffer = nullptr;
    }

    // 5. 初始化各模块算法资源
    if (m_enableUndistort) initUndistortResources(width, height);
    if (m_enableIPM)       initIPMResources(width, height);

    // 6. 预分配 UV 临时变量 (最大尺寸)
    int maxW = std::max(width, m_ipmOutSize.width);
    int maxH = std::max(height, m_ipmOutSize.height);
    m_gpu_u_split.create(maxH/2, maxW/2, CV_8UC1);
    m_gpu_v_split.create(maxH/2, maxW/2, CV_8UC1);
    m_gpu_u_merge.create(maxH/2, maxW/2, CV_8UC1);
    m_gpu_v_merge.create(maxH/2, maxW/2, CV_8UC1);

    return true;
}

// =========================================================================
// [模块 A] 初始化畸变矫正
// =========================================================================
void DecodeThread::initUndistortResources(int width, int height)
{
    m_isUndistortInit = false;
    if (m_xmlPath.isEmpty()) return;

    m_calibParams = loadParamsFromXml(m_xmlPath);
    if (!m_calibParams.isValid) return;

    // 生成 Y Map
    cv::Mat mapX, mapY;
    cv::initUndistortRectifyMap(m_calibParams.K, m_calibParams.D, cv::Mat(),
                                m_calibParams.NewK, cv::Size(width, height),
                                CV_32FC1, mapX, mapY);

    // 生成 UV Map (缩放)
    cv::Mat mapX_UV, mapY_UV;
    cv::resize(mapX, mapX_UV, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_LINEAR);
    cv::resize(mapY, mapY_UV, cv::Size(width / 2, height / 2), 0, 0, cv::INTER_LINEAR);
    mapX_UV = mapX_UV / 2.0f;
    mapY_UV = mapY_UV / 2.0f;

    m_cudaMapX.upload(mapX);
    m_cudaMapY.upload(mapY);
    m_cudaMapX_UV.upload(mapX_UV);
    m_cudaMapY_UV.upload(mapY_UV);

    m_isUndistortInit = true;
    ffDebug("Undistort 资源初始化完成");
}

// =========================================================================
// [模块 B] 初始化 IPM
// =========================================================================
void DecodeThread::initIPMResources(int width, int height)
{
    m_isIPMInit = false;
    if (m_ipmTxtPath.isEmpty()) return;

    std::vector<cv::Point2f> srcPts = loadPointsFromTxt(m_ipmTxtPath);
    if (srcPts.size() != 4) {
        ffDebug("IPM TXT 点数错误: " + QString::number(srcPts.size()));
        return;
    }

    m_ipmParams.srcPoints = srcPts;
    m_ipmParams.outSize = m_ipmOutSize;

    // 1. 目标点 (默认左上、右上、右下、左下)
    std::vector<cv::Point2f> dstPts;
    dstPts.push_back(cv::Point2f(0, 0));
    dstPts.push_back(cv::Point2f(m_ipmOutSize.width, 0));
    dstPts.push_back(cv::Point2f(m_ipmOutSize.width, m_ipmOutSize.height));
    dstPts.push_back(cv::Point2f(0, m_ipmOutSize.height));

    // 2. 计算 Y 通道矩阵
    m_ipmParams.homography = cv::getPerspectiveTransform(srcPts, dstPts);

    // 3. 计算 UV 通道矩阵 (关键！)
    // UV 的分辨率是 Y 的一半，所以源点和目标点坐标都要除以 2
    std::vector<cv::Point2f> srcPtsUV, dstPtsUV;
    for(auto p : srcPts) srcPtsUV.push_back(p * 0.5f);
    for(auto p : dstPts) dstPtsUV.push_back(p * 0.5f);

    m_ipmParams.homographyUV = cv::getPerspectiveTransform(srcPtsUV, dstPtsUV);

    m_isIPMInit = true;
    ffDebug("IPM 资源初始化完成");
}

// =========================================================================
// [执行] 畸变矫正 (Remap)
// =========================================================================
void DecodeThread::executeCudaUndistort(const cv::cuda::GpuMat &srcY, const cv::cuda::GpuMat &srcUV,
                                        cv::cuda::GpuMat &dstY, cv::cuda::GpuMat &dstUV)
{
    if (!m_isUndistortInit) {
        // 如果未初始化但被调用，直接 Copy
        srcY.copyTo(dstY);
        srcUV.copyTo(dstUV);
        return;
    }

    // 1. Y 通道 Remap
    cv::cuda::remap(srcY, dstY, m_cudaMapX, m_cudaMapY, cv::INTER_LINEAR);

    // 2. UV 通道 (Split -> Remap -> Merge)
    // 注意：srcUV 是 NV12 的 UV平面 (Height/2, Width, 2通道)
    // OpenCV GpuMat 不支持 2通道 Remap，所以拆分
    std::vector<cv::cuda::GpuMat> src_split = {m_gpu_u_split, m_gpu_v_split}; // 使用成员变量避免分配

    // 重新调整大小以匹配当前帧 (复用内存)
    // NV12 是 Interleaved U V，但在 OpenCV 里作为 2通道读取
    // 这里我们简单处理：split -> remap 单通道 -> merge

    cv::cuda::split(srcUV, src_split);

    cv::cuda::remap(src_split[0], m_gpu_u_merge, m_cudaMapX_UV, m_cudaMapY_UV, cv::INTER_LINEAR);
    cv::cuda::remap(src_split[1], m_gpu_v_merge, m_cudaMapX_UV, m_cudaMapY_UV, cv::INTER_LINEAR);

    std::vector<cv::cuda::GpuMat> dst_vec = {m_gpu_u_merge, m_gpu_v_merge};
    cv::cuda::merge(dst_vec, dstUV);
}

// =========================================================================
// [执行] 逆透视变换 (WarpPerspective)
// =========================================================================
void DecodeThread::executeCudaIPM(const cv::cuda::GpuMat &srcY, const cv::cuda::GpuMat &srcUV,
                                  cv::cuda::GpuMat &dstY, cv::cuda::GpuMat &dstUV)
{
    if (!m_isIPMInit) {
        srcY.copyTo(dstY);
        srcUV.copyTo(dstUV);
        return;
    }

    // 1. Y 通道变换
    cv::cuda::warpPerspective(srcY, dstY, m_ipmParams.homography, m_ipmParams.outSize);

    // 2. UV 通道变换 (拆分 -> Warp -> 合并)
    std::vector<cv::cuda::GpuMat> src_split = {m_gpu_u_split, m_gpu_v_split};
    cv::cuda::split(srcUV, src_split);

    // 计算 UV 输出尺寸
    cv::Size uvOutSize(m_ipmParams.outSize.width / 2, m_ipmParams.outSize.height / 2);

    cv::cuda::warpPerspective(src_split[0], m_gpu_u_merge, m_ipmParams.homographyUV, uvOutSize);
    cv::cuda::warpPerspective(src_split[1], m_gpu_v_merge, m_ipmParams.homographyUV, uvOutSize);

    std::vector<cv::cuda::GpuMat> dst_vec = {m_gpu_u_merge, m_gpu_v_merge};
    cv::cuda::merge(dst_vec, dstUV);
}

// =========================================================================
// [主循环] GPU 调度
// =========================================================================
void DecodeThread::runGpu()
{
    cudaSetDevice(0);

    while (!stopFlag) {
        FFmpegContext ctx;
        if (!initVideoConnection(ctx)) {
            ffDebug("连接 RTSP 失败，2秒后重试...");
            QThread::msleep(2000); continue;
        }

        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();

        ffDebug("连接成功，开始读取流...");

        while (!stopFlag) {
            // =======================================================
            // [修复] 必须在这里更新“喂狗”时间，否则 3秒后必定超时断开
            // =======================================================
            lastReadTime = std::chrono::high_resolution_clock::now();

            int ret = av_read_frame(ctx.fmtCtx, pkt);
            if (ret < 0) {
                // 打印具体错误码，方便排查是超时(Interrupt)还是断网(EOF)
                char errBuf[128];
                av_strerror(ret, errBuf, sizeof(errBuf));
                ffDebug(QString("流读取中断: %1 (Code: %2)").arg(errBuf).arg(ret));
                break;
            }

            if (pkt->stream_index == ctx.videoStreamIndex) {
                int sendRet = avcodec_send_packet(ctx.codecCtx, pkt);
                if (sendRet == 0) {
                    while (avcodec_receive_frame(ctx.codecCtx, frame) == 0) {

                        int w = frame->width;
                        int h = frame->height;
                        int stride = frame->linesize[0];

                        // 1. 检查资源
                        checkResolutionAndInitResources(w, h, stride);

                        // 2. 准备缓冲区
                        m_bufIdx = 1 - m_bufIdx;
                        uint8_t* finalBuf = m_pBuffers[m_bufIdx];

                        cv::cuda::GpuMat srcY, srcUV;

                        // 3. 封装数据 (区分硬解/软解)
                        if (frame->format == AV_PIX_FMT_CUDA) {
                            srcY = cv::cuda::GpuMat(h, w, CV_8UC1, frame->data[0], stride);
                            srcUV = cv::cuda::GpuMat(h/2, w/2, CV_8UC2, frame->data[1], stride);
                        }
                        else {
                            // 如果是软解码帧，暂时跳过或者打印警告（防止崩溃）
                            // ffDebug("接收到软解码帧，跳过处理...");
                            continue;
                        }

                        // 4. 执行处理流水线
                        if (!srcY.empty()) {
                            // A. 既要矫正，又要 IPM
                            if (m_enableUndistort && m_enableIPM && m_isUndistortInit && m_isIPMInit) {
                                cv::cuda::GpuMat midY(h, w, CV_8UC1, m_pMidBuffer, w);
                                cv::cuda::GpuMat midUV(h/2, w/2, CV_8UC2, m_pMidBuffer + w*h, w);
                                executeCudaUndistort(srcY, srcUV, midY, midUV);

                                int outW = m_ipmParams.outSize.width;
                                int outH = m_ipmParams.outSize.height;
                                cv::cuda::GpuMat dstY(outH, outW, CV_8UC1, finalBuf, outW);
                                cv::cuda::GpuMat dstUV(outH/2, outW/2, CV_8UC2, finalBuf + outW*outH, outW);
                                executeCudaIPM(midY, midUV, dstY, dstUV);
                                emit newFrameDisplayGPU(finalBuf, finalBuf + outW*outH, outW, outH, outW);
                            }
                            // B. 仅矫正
                            else if (m_enableUndistort && m_isUndistortInit) {
                                cv::cuda::GpuMat dstY(h, w, CV_8UC1, finalBuf, w);
                                cv::cuda::GpuMat dstUV(h/2, w/2, CV_8UC2, finalBuf + w*h, w);
                                executeCudaUndistort(srcY, srcUV, dstY, dstUV);
                                emit newFrameDisplayGPU(finalBuf, finalBuf + w*h, w, h, w);
                            }
                            // C. 仅 IPM
                            else if (m_enableIPM && m_isIPMInit) {
                                int outW = m_ipmParams.outSize.width;
                                int outH = m_ipmParams.outSize.height;
                                cv::cuda::GpuMat dstY(outH, outW, CV_8UC1, finalBuf, outW);
                                cv::cuda::GpuMat dstUV(outH/2, outW/2, CV_8UC2, finalBuf + outW*outH, outW);
                                executeCudaIPM(srcY, srcUV, dstY, dstUV);
                                emit newFrameDisplayGPU(finalBuf, finalBuf + outW*outH, outW, outH, outW);
                            }
                            // D. 直通 (拷贝)
                            else {
                                cudaMemcpy2D(finalBuf, w, frame->data[0], stride, w, h, cudaMemcpyDeviceToDevice);
                                cudaMemcpy2D(finalBuf + w*h, w, frame->data[1], stride, w, h/2, cudaMemcpyDeviceToDevice);
                                emit newFrameDisplayGPU(finalBuf, finalBuf + w*h, w, h, w);
                            }
                        }
                    }
                }
            }
            av_packet_unref(pkt);
        }
        releaseVideoResources(ctx);
        av_frame_free(&frame);
        av_packet_free(&pkt);
    }
}
