#ifndef SEGMENTPIC_H
#define SEGMENTPIC_H
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

void getAll_cutpath(vector<vector<int*>>& resultpath, Mat dst, Mat tc_up, bool up, bool debug);
vector<Mat> getpics_with_cutpath(Mat src, Mat dst, int up_left, int down_right, Point up_last, Point down_last, bool debug);
vector<Mat> get_split_imgs_with_skeleton(Mat src, bool debug);

Mat get_FFT_threshold(Mat src, bool debug, string filename, float rotio1, float rotio2, float gamma);
Mat get_FFT_only(Mat src, float rotio1, float rotio2, float gamma);
std::vector<cv::Mat> cutHWLowerNumByH1(cv::Mat img, string filename, bool debug);
std::vector<cv::Mat> cutHWLowerNumByH3(cv::Mat img);
std::vector<cv::Mat> cutHWLowerNumByH4(cv::Mat img1, int num_of_adh, int add_weishu, string filename, bool debug);
std::vector<cv::Mat> cutHWLowerNumByH5(cv::Mat img1, int num_of_adh, bool debug, string filename);


vector<int*> getCoord_by_tessact(Mat gray, tesseract::TessBaseAPI *api);
vector<Mat> get_splitMat(Mat temp);
void removeShortContours(vector<vector<Point>>& contours);
Mat CutNoise_for_split_downnum(Mat img);
#endif // SEGMENTPIC_H

