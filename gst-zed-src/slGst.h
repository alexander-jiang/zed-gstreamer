#include "sl/Camera.hpp"
#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>
#include "NvEglRenderer.h"
#include "nvbuf_utils.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaEGL.h"

// see jetson multimedia api sample app 04_video_dec_trt

// #include "NvCudaProc.h"
// #include "EGL/egl.h"
// #include "EGL/eglext.h"


// DmaBuffer is defined in jetson multimedia api sample app 10_camera_recording/main.cpp
// as "class DmaBuffer : public NvNativeBuffer, public NvBuffer"
// 
// creating NvBuffer example (sample app 05_jpeg_encode):
//         NvBuffer buffer(V4L2_PIX_FMT_YUV420M, ctx.in_width,
//                 ctx.in_height, 0);
//
//         buffer.allocateMemory();
//
// or from jetson_multimedia_api/samples/common/classes/NvJpegDecoder.cpp
//     NvBuffer *out_buf = NULL;
//     out_buf = new NvBuffer(pixel_format, cinfo.image_width,
//             cinfo.image_height, 0);
//     out_buf->allocateMemory();

GstBuffer *slMatToGstBuffer(const sl::Mat &mat)
{
    // EGLImageKHR image = NvEGLImageFromFd(m_eglDisplay, m_fd);
    CUresult status;
    CUeglFrame eglFrame;
    CUgraphicsResource pResource = NULL;
    int ret;
    cudaFree(0);

    // https://docs.nvidia.com/jetson/l4t-multimedia/nvbuf__utils_8h_source.html
    // NvBufferCreate(int *dmabuf_fd, int width, int height,
    //                NvBufferLayout::NvBufferLayout_Pitch, NvBufferColorFormat::NvBufferColorFormat_BGRA_10_10_10_2_2020);
    // maybe we can create the buffer and pass it to gst_buffer_new_wrapped_full

    // status = cuGraphicsEGLRegisterImage(&pResource, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    // if (status != CUDA_SUCCESS)
    // {
    //     printf("cuGraphicsEGLRegisterImage failed : %d \n", status);
    //     return 0;
    // }
    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
    if (status != CUDA_SUCCESS)
    {
        printf("cuGraphicsSubResourceGetMappedArray failed\n");
    }
    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        printf("cuCtxSynchronize failed \n");
    }
    // if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
    //     if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_RGBA) {
    //         cv::cuda::GpuMat mapped(cv::Size(eglFrame.width, eglFrame.height), CV_8UC4,
    //                                 eglFrame.frame.pPitch[0]);
    //         mat.copyTo(mapped);
    //     } else {
    //         printf ("Invalid eglcolorformat for opencv\n");
    //     }
    // }
    // else {
    //     printf ("Invalid frame type for opencv\n");
    // }

    // https://forums.developer.nvidia.com/t/how-to-process-nvbuffer-video-data-in-cuda/61207/10
    // std::vector vec(eglFrame.width * eglFrame.height);
    // cudaMemcpy(&vec[0], eglFrame.frame.pArray[0], vec.size(), cudaMemcpyDeviceToHost);

    sl::uchar1 *pointer = mat.getPtr<sl::uchar1>(sl::MEM::GPU);
    // , mat.getStepBytes(sl::MEM::GPU))
    cudaMemcpy((void *) pointer, eglFrame.frame.pArray[0], mat.getHeight() * mat.getWidth(), cudaMemcpyDeviceToDevice);
    // cudaMalloc((void **)&d_arr, 10 * sizeof(int));


    // create NvBuffer
    int dmabuf_fd;
    NvBufferCreateParams input_params;

    input_params.width = mat.getWidth();
    input_params.height = mat.getHeight();
    input_params.layout = NvBufferLayout_Pitch;
    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.colorFormat = NvBufferColorFormat_NV12;

    ret = NvBufferCreateEx(&dmabuf_fd, &input_params);
    if (ret == -1) {
        printf("NvBufferCreateEx failed \n");
    }

    NvBufferParams nv_buf_params;
    ret = NvBufferGetParams(dmabuf_fd, &nv_buf_params);
    if (ret == -1) {
        printf("NvBufferGetParams failed \n");
    }


    status = cuCtxSynchronize();
    if (status != CUDA_SUCCESS)
    {
        printf("cuCtxSynchronize failed after memcpy \n");
    }
    status = cuGraphicsUnregisterResource(pResource);
    if (status != CUDA_SUCCESS)
    {
        printf("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
    }
    // TODO what is m_params here?? likely NvBufferParams
    // there's a function:
    // int NvBufferGetParams( int dmabuf_fd, NvBufferParams * params) function in nvbuf_utils.h
    // can create the NvBuffer with:
    // int NvBufferCreateEx( int *dmabuf_fd, NvBufferCreateParams *input_params) function in nvbuf_utils.h
    GstBuffer *buffer = gst_buffer_new_wrapped_full(GstMemoryFlags(0),
                                                    nv_buf_params.nv_buffer,
                                                    nv_buf_params.nv_buffer_size, 0,
                                                    nv_buf_params.nv_buffer_size,
                                                    NULL, NULL);
    GstMemory *inmem = gst_buffer_peek_memory(buffer, 0);
    inmem->allocator->mem_type = "nvcam";
    // NvDestroyEGLImage(m_eglDisplay, image);
    return buffer;
}
