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

        try {
            cv::String filename = "./test.png";
            //cv::Mat srcHost = cv::imread(filename, cv::IMREAD_GRAYSCALE);
            cv::Mat gray_image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

            std::chrono::time_point<std::chrono::high_resolution_clock>
                           new_frame_time, new_frame_time2;

            //cv::Mat gray_image;
 	        //cv::cvtColor( srcHost, gray_image, cv::COLOR_BGR2GRAY );   
            //cv::Mat gray_image = cv::Mat(srcHost.rows, srcHost.cols,
            //                    CV_8UC1, srcHost.ptr<uint8_t>(0, 0));

            new_frame_time = std::chrono::high_resolution_clock::now();


            if(TestCUDA) {
                cv::cuda::GpuMat dst, src;
                src.upload(gray_image);

                //cv::cuda::threshold(src,dst,128.0,255.0, CV_THRESH_BINARY);
                cv::cuda::bilateralFilter(src,dst,3,1,1);

                dst.download(gray_image);
            } else {
                cv::bilateralFilter(gray_image,gray_image,3,1,1);
            }

            new_frame_time2 = std::chrono::high_resolution_clock::now();

            cv::imshow("Result",gray_image);
            cv::waitKey(0);

            std::chrono::duration<double> duration(new_frame_time2 - new_frame_time);
            std::cout << "filter time:" << duration.count()*1000.0 << "ms" << std::endl;

        } catch(const cv::Exception& ex) {
            std::cout << "Error: " << ex.what() << std::endl;
        }

}
