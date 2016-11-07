remap : test.cu
	nvcc --compiler-bindir /usr/bin -O2 -o remap test.cu `pkg-config --cflags opencv` `pkg-config --libs opencv`

clean:
	rm -f remap
#cgdb ./remap

#g++ -o remap Remap_Demo.cpp -I${OPENCV_DIR}/include -L${OPENCV_DIR}/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

