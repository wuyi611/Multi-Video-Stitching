#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QMutex>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class VideoOpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit VideoOpenGLWidget(QWidget *parent = nullptr);
    ~VideoOpenGLWidget();

public slots:
    // 接收 GPU 显存中的 NV12 数据
    void updateImageGPU(uint8_t* y, uint8_t* uv, int w, int h, int stride);

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

private:
    void setupShaders();
    void initTextures();
    void freeResources(); // 统一释放资源

private:
    QMutex m_dataMutex;

    // 视频数据参数
    uint8_t *m_yPtr = nullptr;   // GPU 指针
    uint8_t *m_uvPtr = nullptr;  // GPU 指针
    int m_imgW = 0;
    int m_imgH = 0;
    int m_stride = 0;

    // OpenGL 资源
    QOpenGLShaderProgram *m_program = nullptr;
    GLuint m_texY = 0;
    GLuint m_texUV = 0;

    // 记录纹理当前尺寸 (用于判断是否需要重建)
    int m_texW = 0;
    int m_texH = 0;

    // CUDA-GL 互操作资源句柄
    cudaGraphicsResource* m_cudaResY = nullptr;
    cudaGraphicsResource* m_cudaResUV = nullptr;
};
