#include <stdexcept>
#include <string>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

using namespace std;
using namespace cv;

int main() {

  Mat src, dst, xmap, ymap;
  cuda::GpuMat d_src, d_dst, d_xmap, d_ymap;

  int interpolation = INTER_LINEAR;
  int borderMode = BORDER_REPLICATE;

  for (int size = 1000; size <= 4000; size *= 2)
  {

    src = imread("Car-Wallpaper-HD-1080x1920-4.jpg", CV_LOAD_IMAGE_COLOR);
    int nrow = src.rows;
    int ncol = src.cols;
    printf("nrow: %d, ncol: %d\n", nrow, ncol);

    xmap.create(size, size, CV_32F);
    ymap.create(size, size, CV_32F);
    for (int i = 0; i < size; ++i)
    {
      float* xmap_row = xmap.ptr<float>(i);
      float* ymap_row = ymap.ptr<float>(i);
      for (int j = 0; j < size; ++j)
      {
        xmap_row[j] = (j - size * 0.5f) * 0.75f + size * 0.5f;
        ymap_row[j] = (i - size * 0.5f) * 0.75f + size * 0.5f;
      }
    }

    remap(src, dst, xmap, ymap, interpolation, borderMode);

    char fname[256];
    std::sprintf(fname, "output_%d.jpg", size);
    imwrite(fname, dst);

    d_src.upload(src);
    d_xmap.upload(xmap);
    d_ymap.upload(ymap);

    cuda::remap(d_src, d_dst, d_xmap, d_ymap, interpolation, borderMode);

    Mat cudst;
    d_dst.download(cudst);
    std::sprintf(fname, "cuoutput_%d.jpg", size);
    imwrite(fname, cudst);

  }

  return 0;
}
