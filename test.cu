#include <stdexcept>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/cuda.hpp"
//#include "opencv2/core.hpp"
//#include "opencv2/core/cuda_types.hpp"
//#include "opencv2/core/cuda.inl.hpp"
#include <sys/time.h>

using namespace std;
using namespace cv;

int main() {

  Mat dst;
  cuda::GpuMat d_src, d_dst, d_xmap, d_ymap;

  int interpolation = INTER_LINEAR;
  int borderMode = BORDER_REPLICATE;

  cudaSetDevice(1);

  int gpucount = cuda::getCudaEnabledDeviceCount();
  printf("gpucount: %d\n", gpucount);

  Mat _src = imread("Car-Wallpaper-HD-1080x1920-4.jpg", CV_LOAD_IMAGE_COLOR);
  cuda::HostMem src(_src, cuda::HostMem::PAGE_LOCKED);

  for (int size = 1000; size <= 4000; size *= 2)
  {

    int nrow = src.rows;
    int ncol = src.cols;
    printf("nrow: %d, ncol: %d\n", nrow, ncol);

    cuda::HostMem xmap(size, size, CV_32F, cuda::HostMem::PAGE_LOCKED);
    cuda::HostMem ymap(size, size, CV_32F, cuda::HostMem::PAGE_LOCKED);
    Mat h_xmap = xmap.createMatHeader();
    Mat h_ymap = xmap.createMatHeader();

    for (int i = 0; i < size; ++i)
    {
      float* xmap_row = h_xmap.ptr<float>(i);
      float* ymap_row = h_ymap.ptr<float>(i);
      for (int j = 0; j < size; ++j)
      {
        xmap_row[j] = (j - size * 0.5f) * 0.75f + size * 0.5f;
        ymap_row[j] = (i - size * 0.5f) * 0.75f + size * 0.5f;
      }
    }

    struct timeval start, stop;
    gettimeofday(&start, NULL);
    remap(_src, dst, h_xmap, h_ymap, interpolation, borderMode);
    gettimeofday(&stop, NULL);
    float elapse = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) * 1e-6;
    printf("cpu remap time:               %f\n", elapse);

    char fname[256];
    std::sprintf(fname, "output_%d.jpg", size);
    imwrite(fname, dst);

    d_xmap.upload(xmap);
    d_ymap.upload(ymap);

    gettimeofday(&start, NULL);
    d_src.upload(src);
    gettimeofday(&stop, NULL);
    elapse = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) * 1e-6;
    printf("gpu upload time:              %f\n", elapse);

    gettimeofday(&start, NULL);
    cuda::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);
    gettimeofday(&stop, NULL);
    elapse = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) * 1e-6;
    printf("gpu remap  time:              %f\n", elapse);

    cuda::HostMem cudst(size, size, src.type(), cuda::HostMem::PAGE_LOCKED);
    gettimeofday(&start, NULL);
    d_dst.download(cudst.createMatHeader());
    gettimeofday(&stop, NULL);
    elapse = stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) * 1e-6;
    printf("gpu download time:            %f\n", elapse);

    std::sprintf(fname, "cuoutput_%d.jpg", size);
    imwrite(fname, cudst.createMatHeader());

  }

  return 0;
}
