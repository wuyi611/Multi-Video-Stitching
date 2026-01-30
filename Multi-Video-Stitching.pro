QT       += core gui openglwidgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    DecodeThread.cpp \
    VideoOpenGLWidget.cpp \
    main.cpp \
    Widget.cpp

HEADERS += \
    DecodeThread.h \
    VideoOpenGLWidget.h \
    Widget.h

FORMS += \
    Widget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    res.qrc


# FFmpeg8.0
LIBS += -L$$PWD/ffmpeg-8.0-full_build-shared/lib/ -lavcodec -lavdevice -lavfilter -lavformat -lavutil -lswresample -lswscale

INCLUDEPATH += $$PWD/ffmpeg-8.0-full_build-shared/include
DEPENDPATH += $$PWD/ffmpeg-8.0-full_build-shared/include


# cuda12.6
LIBS += -L$$PWD/cuda12.6/lib/x64 -lcuda -lcudart
INCLUDEPATH += $$PWD/cuda12.6/include
DEPENDPATH += $$PWD/cuda12.6/include

