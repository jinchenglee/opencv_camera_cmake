#include <iostream>
#include <ctime>
#include <cmath>
#include "bits/time.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <chrono>
#include <ctime>

#include <stdio.h>

#define TestCUDA true

int main() {
    std::clock_t begin = std::clock();

        try {
            //cv::String filename = "./test.png";
            //cv::Mat srcHost = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            cv::Mat srcHost;
            cv::Mat resultHost;
            cv::VideoCapture cap;

            cap.open(0); // Open first camera.

            // Set camera parameters.
            cap.set(cv::CAP_PROP_FORMAT, -1);  // turn off video decoder (extract stream)

            double fps2 = cap.get(cv::CAP_PROP_FPS);
            std::cout << "fps2 : " << fps2 << std::endl;

        int fpsCamera = 30;
        std::chrono::time_point<std::chrono::high_resolution_clock>
                   prev_frame_time(std::chrono::high_resolution_clock::now());
        std::chrono::time_point<std::chrono::high_resolution_clock>
                           new_frame_time, new_frame_time2;

            for(int i=0; i<1000; i++) {

                cap.read(srcHost);

			if (i==10) {
				FILE *fout = fopen("frame.raw", "wb");
				fwrite((void*)srcHost.data, 1504*480, 1, fout);
				fclose(fout);
			}



                std::cout << "rows,cols,size,type,elemSize=" << srcHost.rows << "," << srcHost.cols 
                        << "," << srcHost.total() << "," << srcHost.type() << "," 
                        << srcHost.elemSize() << std::endl;

                //cv::Mat gray_image;
 	            //cv::cvtColor( srcHost, gray_image, cv::COLOR_BGR2GRAY );   
                //cv::Mat gray_image = cv::Mat(srcHost.rows, srcHost.cols,
                //                    CV_8UC1, srcHost.ptr<uint8_t>(0, 0));
                //cv::Mat myimages[3];
                //cv::split(srcHost, myimages);//splitting images into 3 different channels. 


                cv::imshow("Result",srcHost); // Only show the 1st channel.
                cv::waitKey(1);

            new_frame_time2 = std::chrono::high_resolution_clock::now();

                if(TestCUDA) {
                    cv::cuda::GpuMat dst, src;
                    src.upload(srcHost);

                    //cv::cuda::threshold(src,dst,128.0,255.0, CV_THRESH_BINARY);
                    cv::cuda::bilateralFilter(src,dst,3,1,1);

                    dst.download(resultHost);
                } else {
                    cv::bilateralFilter(srcHost,resultHost,3,1,1);
                }

                //cv::imshow("Result",resultHost);

            new_frame_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration1(new_frame_time - prev_frame_time);
            double fps = 1/duration1.count();
            std::cout << "fps : " << fps << std::endl;

            std::chrono::duration<double> duration2(new_frame_time - new_frame_time2);
            std::cout << "filter time:" << duration2.count()*1000.0 << "ms" << std::endl;
            std::cout << "total time:" << duration1.count()*1000.0 << "ms" << std::endl;

                prev_frame_time = new_frame_time;
                  
                //cv::imshow("Result",resultHost);
                //cv::waitKey(1);
            }

            cap.release();

        } catch(const cv::Exception& ex) {
            std::cout << "Error: " << ex.what() << std::endl;
        }

    std::clock_t end = std::clock();
    std::cout << double(end-begin) / CLOCKS_PER_SEC  << std::endl;
}
