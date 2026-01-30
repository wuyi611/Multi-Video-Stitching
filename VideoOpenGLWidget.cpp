#include "VideoOpenGLWidget.h"
#include <QDebug>

VideoOpenGLWidget::VideoOpenGLWidget(QWidget *parent) : QOpenGLWidget(parent) {}

VideoOpenGLWidget::~VideoOpenGLWidget() {
    makeCurrent();
    freeResources(); // 析构时释放所有资源
    delete m_program;
    doneCurrent();
}

void VideoOpenGLWidget::freeResources() {
    // 1. 注销 CUDA 资源
    if (m_cudaResY) {
        cudaGraphicsUnregisterResource(m_cudaResY);
        m_cudaResY = nullptr;
    }
    if (m_cudaResUV) {
        cudaGraphicsUnregisterResource(m_cudaResUV);
        m_cudaResUV = nullptr;
    }
    // 2. 删除 OpenGL 纹理
    if (m_texY) {
        glDeleteTextures(1, &m_texY);
        m_texY = 0;
    }
    if (m_texUV) {
        glDeleteTextures(1, &m_texUV);
        m_texUV = 0;
    }
}

void VideoOpenGLWidget::setupShaders() {
    m_program = new QOpenGLShaderProgram(this);

    // 顶点着色器 (保持不变)
    const char *vsrc = R"(
        attribute vec4 vertexIn;
        attribute vec2 textureIn;
        varying vec2 textureOut;
        void main(void) {
            gl_Position = vertexIn;
            textureOut = textureIn;
        }
    )";

    // 片元着色器 (保持不变，负责 NV12 -> RGB)
    const char *fsrc = R"(
        varying vec2 textureOut;
        uniform sampler2D tex_y;
        uniform sampler2D tex_uv;
        void main(void) {
            vec3 yuv;
            vec3 rgb;
            yuv.x = texture2D(tex_y, textureOut).r;
            yuv.y = texture2D(tex_uv, textureOut).r - 0.5;
            yuv.z = texture2D(tex_uv, textureOut).a - 0.5;
            rgb.r = yuv.x + 1.402 * yuv.z;
            rgb.g = yuv.x - 0.34414 * yuv.y - 0.71414 * yuv.z;
            rgb.b = yuv.x + 1.772 * yuv.y;
            gl_FragColor = vec4(rgb, 1.0);
        }
    )";

    m_program->addShaderFromSourceCode(QOpenGLShader::Vertex, vsrc);
    m_program->addShaderFromSourceCode(QOpenGLShader::Fragment, fsrc);
    m_program->bindAttributeLocation("vertexIn", 0);
    m_program->bindAttributeLocation("textureIn", 1);
    m_program->link();
}

void VideoOpenGLWidget::initializeGL() {
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    setupShaders();
    // 纹理生成移到了 paintGL 里动态处理
}

void VideoOpenGLWidget::updateImageGPU(uint8_t* y, uint8_t* uv, int w, int h, int stride) {
    QMutexLocker locker(&m_dataMutex);
    if (!y || !uv || w <= 0 || h <= 0) return;
    m_yPtr = y;
    m_uvPtr = uv;
    m_imgW = w;
    m_imgH = h;
    m_stride = stride;
    update();
}

void VideoOpenGLWidget::paintGL() {
    QMutexLocker locker(&m_dataMutex);

    // 如果没有数据或 OpenGL 上下文未就绪，清屏退出
    if (!m_yPtr || !m_uvPtr || m_imgW == 0 || m_imgH == 0) {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        return;
    }

    // =============================================================
    // 1. 资源初始化 (如果分辨率改变或首次运行)
    // =============================================================
    if (m_texW != m_imgW || m_texH != m_imgH) {
        // 先清理旧资源
        freeResources();

        m_texW = m_imgW;
        m_texH = m_imgH;

        // --- 创建 Y 纹理 ---
        glGenTextures(1, &m_texY);
        glBindTexture(GL_TEXTURE_2D, m_texY);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // 分配显存空间 (注意：data 传 nullptr，只占坑)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, m_texW, m_texH, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // [关键] 注册 Y 纹理给 CUDA
        cudaGraphicsGLRegisterImage(&m_cudaResY, m_texY, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

        // --- 创建 UV 纹理 ---
        glGenTextures(1, &m_texUV);
        glBindTexture(GL_TEXTURE_2D, m_texUV);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        // UV 纹理宽高各减半，双通道 (LUMINANCE_ALPHA)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE_ALPHA, m_texW / 2, m_texH / 2, 0, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        // [关键] 注册 UV 纹理给 CUDA
        cudaGraphicsGLRegisterImage(&m_cudaResUV, m_texUV, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    }

    // =============================================================
    // 2. CUDA-GL 互操作：将数据从 m_yPtr (解码显存) 拷贝到 纹理 (渲染显存)
    // =============================================================
    if (m_cudaResY && m_cudaResUV) {
        // A. 映射 (Lock)
        cudaGraphicsResource* resources[2] = {m_cudaResY, m_cudaResUV};
        cudaGraphicsMapResources(2, resources, 0);

        // B. 获取纹理对应的 CUDA 数组指针
        cudaArray_t arrayY, arrayUV;
        cudaGraphicsSubResourceGetMappedArray(&arrayY, m_cudaResY, 0, 0);
        cudaGraphicsSubResourceGetMappedArray(&arrayUV, m_cudaResUV, 0, 0);

        // C. 执行拷贝 (Device to Array) - 全程 GPU 内部操作，极快！
        // 这里的 m_yPtr 和 m_uvPtr 是从 DecodeThread 传来的 GPU 指针
        cudaMemcpy2DToArray(arrayY, 0, 0, m_yPtr, m_stride, m_imgW, m_imgH, cudaMemcpyDeviceToDevice);
        cudaMemcpy2DToArray(arrayUV, 0, 0, m_uvPtr, m_stride, m_imgW, m_imgH / 2, cudaMemcpyDeviceToDevice);

        // D. 解除映射 (Unlock)
        cudaGraphicsUnmapResources(2, resources, 0);
    }

    // =============================================================
    // 3. OpenGL 纯着色器渲染 (NV12 -> RGB)
    // =============================================================

    // 清屏
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 计算保持比例的 Viewport
    float ratioW = (float)width() / m_imgW;
    float ratioH = (float)height() / m_imgH;
    float r = std::min(ratioW, ratioH);
    int vw = (int)(m_imgW * r);
    int vh = (int)(m_imgH * r);
    glViewport((width() - vw) / 2, (height() - vh) / 2, vw, vh);

    m_program->bind();

    // 绑定 Y 纹理到单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texY);
    m_program->setUniformValue("tex_y", 0);

    // 绑定 UV 纹理到单元 1
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_texUV);
    m_program->setUniformValue("tex_uv", 1);

    // 绘制全屏四边形
    static const GLfloat vertices[] = { -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f };
    static const GLfloat texCoords[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };

    m_program->enableAttributeArray(0);
    m_program->setAttributeArray(0, GL_FLOAT, vertices, 2);
    m_program->enableAttributeArray(1);
    m_program->setAttributeArray(1, GL_FLOAT, texCoords, 2);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    m_program->disableAttributeArray(0);
    m_program->disableAttributeArray(1);
    m_program->release();

    // 解绑纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void VideoOpenGLWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}
