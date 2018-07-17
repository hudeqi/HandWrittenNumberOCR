#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <QString>
#include <QDir>
#include <QFile>
#include "segmentpic.h"
#include "cutcol.h"
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
#define min_length_for_num 5  //10
#define skeleton_debug false


void getAll_cutpath(vector<vector<int*>>& resultpath, Mat dst, Mat tc_up, bool up, bool debug) {
    vector<int*> backcoord;
    for (int i = 0; i < dst.rows; i++)
        for (int j = 0; j < dst.cols; j++) {
            if ((int)dst.at<uchar>(i, j) == 255) {
                int* a = new int[2];
                a[0] = i;
                a[1] = j;
                backcoord.push_back(a);
            }
        }
    sort(backcoord.begin(), backcoord.end(), contoursizeCmp);
    int left = backcoord[0][1] * 0.85 + 0.15 * backcoord[backcoord.size() - 1][1];
    int right = backcoord[backcoord.size() - 1][1] * 0.85 + 0.15 * backcoord[0][1];
    int lefth = gethighestpoint_x(backcoord, left, right)[0];
    int righth = gethighestpoint_x(backcoord, left, right)[1];
    int tempx = lefth;
    int tempy = left;
    line(dst, Point(left - 2, 0), Point(left - 2, dst.rows - 1), Scalar(0));
    line(dst, Point(right + 2, 0), Point(right + 2, dst.rows - 1), Scalar(0));
    /*line(dst, Point(left - 2, 0), Point(left - 2, lefth + 5), Scalar(0));
    line(dst, Point(right + 2, 0), Point(right + 2, righth + 5), Scalar(0));
    dst.at<uchar>(lefth + 5, left - 2) = 255;
    dst.at<uchar>(righth + 5, right + 2) = 255;*/

    vector<vector<int*>> temppathlist;
    vector<pair<int*, vector<int*>>> tolookpoint;
    vector<int*> cutpath;
    vector<int*> passed_allpoints;
    vector<int> temp_all_num;
    int* crosspoint = new int[2];
    int* a = new int[2];
    a[0] = tempx;
    a[1] = tempy;
    cutpath.push_back(a);
    passed_allpoints.push_back(a);
    vector<int*> neighbor = getneighbor(dst, tempx, tempy);
    vector<int*> filter_neighbor;
    while (neighbor.size() != 1) {
        for (int n = 0; n < neighbor.size(); n++) {
            if (!(existin_passed_points(neighbor[n], passed_allpoints))) {
                filter_neighbor.push_back(neighbor[n]);
            }
        }
        if (filter_neighbor.size() <= 1) {
            if (filter_neighbor.size() == 1) {
                tempx = filter_neighbor[0][0];
                tempy = filter_neighbor[0][1];
                int* a = new int[2];
                a[0] = tempx;
                a[1] = tempy;
                cutpath.push_back(a);
                passed_allpoints.push_back(a);
            }
            else {
                break;
            }
        lookdown:
            neighbor = getneighbor(dst, tempx, tempy);

        }
        else if (filter_neighbor.size() > 1) {
            crosspoint[0] = tempx;
            crosspoint[1] = tempy;
            tolookpoint.push_back(make_pair(crosspoint, filter_neighbor));
            temp_all_num.push_back(filter_neighbor.size());
            for (int j = 0; j < filter_neighbor.size() - 1; j++) {
                vector<int*> newpath;
                for (int i = 0; i < cutpath.size(); i++) {
                    newpath.push_back(cutpath[i]);
                }
                temppathlist.push_back(newpath);
            }
        nextpoint:
            while (tolookpoint[tolookpoint.size() - 1].second.size() > 0) {
                tempx = tolookpoint[tolookpoint.size() - 1].second[tolookpoint[tolookpoint.size() - 1].second.size() - 1][0];
                tempy = tolookpoint[tolookpoint.size() - 1].second[tolookpoint[tolookpoint.size() - 1].second.size() - 1][1];
                int* a = new int[2];
                a[0] = tempx;
                a[1] = tempy;
                cutpath.push_back(a);
                passed_allpoints.push_back(a);
                if (tolookpoint[tolookpoint.size() - 1].second.size() < (temp_all_num[temp_all_num.size() - 1])) {
                    temppathlist.pop_back();
                }
                tolookpoint[tolookpoint.size() - 1].second.pop_back();

                Mat tt = tc_up.clone();
                for (int m = 0; m < cutpath.size(); m++) {
                    tt.at<uchar>(cutpath[m][0], cutpath[m][1]) = 255;
                }
                if (debug){
                    imshow("fc", tt);
                    waitKey();
                }
                goto lookdown;
            }
            tolookpoint.pop_back();
            temp_all_num.pop_back();
            goto nextpoint;
        }
        filter_neighbor.clear();
    }

    if (cutpath.size() > 0) {
        resultpath.push_back(cutpath);
    }
    if (debug){
        imshow("1", dst);
    }
    if (tolookpoint[tolookpoint.size() - 1].second.size() > 0) {
        Mat temppic = tc_up.clone();
        for (int j = 0; j < cutpath.size(); j++) {
            temppic.at<uchar>(cutpath[j][0], cutpath[j][1]) = 255;
        }
        if (debug){
            imshow("2", temppic);
            waitKey();
        }
        cutpath.clear();
        cutpath = temppathlist[temppathlist.size() - 1];
        goto nextpoint;
    }
    else {
        if ((tolookpoint.size() - 1) > 0) {
            tolookpoint.pop_back();
            temp_all_num.pop_back();
            cutpath.clear();
            if (temppathlist.size() > 0) {
                cutpath = temppathlist[temppathlist.size() - 1];
                goto nextpoint;
            }
        }
    }
}

vector<Mat> getpics_with_cutpath(Mat src, Mat dst, int up_left, int down_right, Point up_last, Point down_last, bool debug) {
    vector<Mat> result;
    Mat skeleton = dst - src;
    line(skeleton, Point(0, 0), Point(up_left, 0), Scalar(255));
    line(skeleton, Point(0, 0), Point(0, skeleton.rows - 1), Scalar(255));
    line(skeleton, Point(0, skeleton.rows - 1), Point(down_right, skeleton.rows - 1), Scalar(255));
    line(skeleton, up_last, down_last, Scalar(255));
    if (debug){
        imshow("ske", skeleton);
    }
    IplImage temp = IplImage(skeleton);
    IplImage *img = &temp;
    cvFloodFill(img, cvPoint(1, 1), cvScalarAll(255));
    Mat mask(img);
    if (debug){
        imshow("mask", mask);
    }
    Mat result1 = src.clone();
    Mat result2 = src.clone();
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if ((int)mask.at<uchar>(i, j) == 0) {
                result1.at<uchar>(i, j) = 0;
            }
        }
    }
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            if ((int)mask.at<uchar>(i, j) == 255) {
                result2.at<uchar>(i, j) = 0;
            }
        }
    }
    if (debug){
        imshow("result1", result1);
        imshow("result2", result2);
        waitKey();
    }
    result.push_back(result1);
    result.push_back(result2);
    return result;
}

vector<Mat> get_split_imgs_with_skeleton(Mat src, bool debug){
    Mat tempsrc = src.clone();
    copyMakeBorder(tempsrc, tempsrc, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
    Mat tc_up = tempsrc.clone();
    Mat tc_down = tempsrc.clone();
    Mat tc_result = tempsrc.clone();
    Mat tocut = tempsrc.clone();
    flip(tc_down, tc_down, 0);
    flip(tc_down, tc_down, 1);
    IplImage tempic = IplImage(tempsrc);
    IplImage *img = &tempic;
    CvMemStorage * storage = cvCreateMemStorage(0);
    CvSeq *contours = 0;
    cvFindContours(img, storage, &contours, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    cvDrawContours(img, contours, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), -1, CV_FILLED, 8);
    Mat s1(img);
    cvtColor(s1, s1, CV_GRAY2BGR);
    Mat dst = getthinImage(s1);
    Mat dstclone_for_crosspoint = dst.clone();
    flip(dstclone_for_crosspoint, dstclone_for_crosspoint, 0);
    flip(dstclone_for_crosspoint, dstclone_for_crosspoint, 1);
    dst = dst * 255;
    dstclone_for_crosspoint = dstclone_for_crosspoint * 255;

    vector<vector<int*>> resultpath_up;
    vector<vector<int*>> resultpath_down;
    getAll_cutpath(resultpath_up, dst, tc_up, true, debug);
    getAll_cutpath(resultpath_down, dstclone_for_crosspoint, tc_down, false, debug);

    for (int i = 0; i < resultpath_down.size(); i++) {
        //for (int j = 0; j < resultpath_down[i].size(); j++) {
        resultpath_down[i][resultpath_down[i].size() - 1][0] = dst.rows - 1 - resultpath_down[i][resultpath_down[i].size() - 1][0];
        resultpath_down[i][resultpath_down[i].size() - 1][1] = dst.cols - 1 - resultpath_down[i][resultpath_down[i].size() - 1][1];
        //}
    }
    int result_uppath_id = 0;
    int result_downpath_id = 0;
    int mindistance = 9999999;
    for (int i = 0; i < resultpath_up.size(); i++) {
        for (int j = 0; j < resultpath_down.size(); j++) {
            int distance = pow(resultpath_up[i][resultpath_up[i].size() - 1][0] - resultpath_down[j][resultpath_down[j].size() - 1][0], 2)
                + pow(resultpath_up[i][resultpath_up[i].size() - 1][1] - resultpath_down[j][resultpath_down[j].size() - 1][1], 2);
            if (distance < mindistance) {
                mindistance = distance;
                result_uppath_id = i;
                result_downpath_id = j;
            }
        }
    }
    for (int j = 0; j < resultpath_down[result_downpath_id].size() - 1; j++) {
        resultpath_down[result_downpath_id][j][0] = dst.rows - 1 - resultpath_down[result_downpath_id][j][0];
        resultpath_down[result_downpath_id][j][1] = dst.cols - 1 - resultpath_down[result_downpath_id][j][1];
    }
    for (int i = 0; i < resultpath_up[result_uppath_id].size(); i++) {
        tc_result.at<uchar>(resultpath_up[result_uppath_id][i][0], resultpath_up[result_uppath_id][i][1]) = 255;
    }
    if (debug){
        imshow("vfdvgb2", tc_result);
    }
    for (int i = 0; i < resultpath_down[result_downpath_id].size(); i++) {
        tc_result.at<uchar>(resultpath_down[result_downpath_id][i][0], resultpath_down[result_downpath_id][i][1]) = 255;
    }
    if (debug){
        imshow("vfdvgb3", tc_result);
    }
    line(tc_result, Point(resultpath_up[result_uppath_id][resultpath_up[result_uppath_id].size() - 1][1],
        resultpath_up[result_uppath_id][resultpath_up[result_uppath_id].size() - 1][0]),
        Point(resultpath_down[result_downpath_id][resultpath_down[result_downpath_id].size() - 1][1],
            resultpath_down[result_downpath_id][resultpath_down[result_downpath_id].size() - 1][0]), Scalar(255));
    if (debug){
        imshow("vfdvgb4", tc_result);
    }
    line(tc_result, Point(resultpath_up[result_uppath_id][0][1], 0), Point(resultpath_up[result_uppath_id][0][1], resultpath_up[result_uppath_id][0][0]), Scalar(255));
    if (debug){
        imshow("vfdvgb5", tc_result);
    }
    line(tc_result, Point(resultpath_down[result_downpath_id][0][1], tc_result.rows - 1), Point(resultpath_down[result_downpath_id][0][1], resultpath_down[result_downpath_id][0][0]), Scalar(255));
    if (debug){
        imshow("vfdvgb", tc_result);
        waitKey();
    }

    vector<Mat> split_imgs = getpics_with_cutpath(tocut, tc_result, resultpath_up[result_uppath_id][0][1], resultpath_down[result_downpath_id][0][1],
        Point(resultpath_up[result_uppath_id][resultpath_up[result_uppath_id].size() - 1][1],
        resultpath_up[result_uppath_id][resultpath_up[result_uppath_id].size() - 1][0]),
        Point(resultpath_down[result_downpath_id][resultpath_down[result_downpath_id].size() - 1][1],
            resultpath_down[result_downpath_id][resultpath_down[result_downpath_id].size() - 1][0]), debug);
    return split_imgs;
}




Mat get_FFT_threshold(Mat src, bool debug, string filename, float rotio1, float rotio2, float gamma){
    Mat sgray;
    cvtColor(src, sgray, CV_BGR2GRAY);
    float sum = 0.0f;
    for (int i = 0; i < sgray.rows; i++)
        for (int j = 0; j < sgray.cols; j++) {
            sum = sum + ((int)sgray.at<uchar>(i, j) / 255.0);
        }
    float whiterotio = sum / sgray.rows / sgray.cols;
    float b = 6.5;
    if (whiterotio > rotio1){//0.855,0.8       后面是给tessact定位用的FFT参数
        b = 6.5;//6.5
    }
    else if(whiterotio <= rotio1 && whiterotio > rotio2){//0.855,0.735,0.8,0.7
        b = 7.0;//7.0
    }
    else{
        b = 7.0;//7.0
    }

    Mat dst, input, gray;
    Mat img2 = src.clone();
    Mat BGR[3];
    split(img2, BGR);
    BGR[0] = mask(BGR[0], b, b);
    BGR[1] = mask(BGR[1], b, b);
    BGR[2] = mask(BGR[2], b, b);
    Mat maskBGR;
    merge(BGR, 3, maskBGR);
    maskBGR = maskBGR(Rect(0, 0, img2.size().width, img2.size().height));
    cout << maskBGR.size() << maskBGR.type();
    cout << img2.size() << img2.type();
    img2.convertTo(img2, maskBGR.type(), 2.4 / 255.0);
    //imshow("maskBGR", maskBGR);
    maskBGR = img2 - maskBGR + 1.0 / 255.0;
    //namedWindow("InputImage");
    //imshow("InputImage", img2);
    //imshow("RESULT", maskBGR);
    vector<Mat> channels;
    split(maskBGR, channels);
    Mat red = channels[2];
    if (debug){
    imshow("red", red);}
    imwrite("chn-p4-red.png", 255 * red);
    input = imread("chn-p4-red.png");
    cvtColor(input, gray, CV_BGR2GRAY);
    float fGamma = gamma; //////////////////////////// 5.0,2.2   后面是给tessact定位用的FFT参数
    MyGammaCorrection(gray, dst, fGamma);
    if (debug){
    imshow("Dst", dst);}

    char save_file_name[300];
    //string s="/home/ubuntu/下载/500_check_data/smallNum/原图框小写金额/gray1/" + filename;
    string s="/home/ubuntu/桌面/图片/right_gray/" + filename;
    string s2=s+"_%d.png";
    sprintf(save_file_name,s.c_str(), 0);
    imwrite(save_file_name,dst);


    threshold(dst, dst, 0, 255, THRESH_OTSU);
    bitwise_not(dst, dst);
    RemoveSmallRegion(dst, dst, 5, 1, 1);
    if (debug){
        imshow("thres", dst);
        waitKey();
    }
    return dst;
}

Mat get_FFT_only(Mat src, float rotio1, float rotio2, float gamma){
    Mat sgray;
    cvtColor(src, sgray, CV_BGR2GRAY);
    float sum = 0.0f;
    for (int i = 0; i < sgray.rows; i++)
        for (int j = 0; j < sgray.cols; j++) {
            sum = sum + ((int)sgray.at<uchar>(i, j) / 255.0);
        }
    float whiterotio = sum / sgray.rows / sgray.cols;
    float b = 6.5;
    if (whiterotio > rotio1){//0.855,0.8       后面是给tessact定位用的FFT参数
        b = 6.5;//6.5
    }
    else if(whiterotio <= rotio1 && whiterotio > rotio2){//0.855,0.735,0.8,0.7
        b = 7.0;//7.0
    }
    else{
        b = 7.0;//7.0
    }

    Mat dst, input, gray;
    Mat img2 = src.clone();
    Mat BGR[3];
    split(img2, BGR);
    BGR[0] = mask(BGR[0], b, b);
    BGR[1] = mask(BGR[1], b, b);
    BGR[2] = mask(BGR[2], b, b);
    Mat maskBGR;
    merge(BGR, 3, maskBGR);
    maskBGR = maskBGR(Rect(0, 0, img2.size().width, img2.size().height));
    cout << maskBGR.size() << maskBGR.type();
    cout << img2.size() << img2.type();
    img2.convertTo(img2, maskBGR.type(), 2.4 / 255.0);
    //imshow("maskBGR", maskBGR);
    maskBGR = img2 - maskBGR + 1.0 / 255.0;
    //namedWindow("InputImage");
    //imshow("InputImage", img2);
    //imshow("RESULT", maskBGR);
    vector<Mat> channels;
    split(maskBGR, channels);
    Mat red = channels[2];
    imwrite("chn-p4-red.png", 255 * red);
    input = imread("chn-p4-red.png");
    cvtColor(input, gray, CV_BGR2GRAY);
    float fGamma = gamma; //////////////////////////// 5.0,2.2   后面是给tessact定位用的FFT参数
    MyGammaCorrection(gray, dst, fGamma);

    return dst;
}




std::vector<cv::Mat> cutHWLowerNumByH1(Mat img1, string filename, bool debug){
    vector<Mat> v;
//    string ff = fileName.substr(fileName.length() - 1, fileName.length());
//    if (ff.compare("g") == 0) {
//        cout << "file name:" << fileName << endl;
//        Mat img1;
//        Mat img0 = imread(fileName);
//        cvtColor(img0, img1, CV_RGB2GRAY);
//        threshold(img1, img1, 0, 255, CV_THRESH_OTSU);

        Mat img1_cutline = img1.clone();
        bitwise_not(img1, img1);
//        RemoveSmallRegion(img1, img1, 50, 1, 1);
        //imshow("hy", img1);
        vector<vector<Point>> contours1;
        Mat img1clone = img1.clone();
        findContours(img1,
                     contours1,
                     CV_RETR_EXTERNAL,
                     //CV_RETR_TREE,
                     CV_CHAIN_APPROX_NONE);
        //hu2月7号添加 噪声去除
        vector<vector<Point>> contours = removeShortContoursfornumber(contours1, img1.rows);
        //remove_vertical_line_between_number(contours, 0.45);
//        Mat contoursImg = Mat(img1.rows, img1.cols, CV_8U);
//            drawContours(contoursImg, contours);
//            imshow("contoursImg", contoursImg);
//            waitKey();
        vector<int*> coordinates = CCPoints2Coordinates1(contours);
        sort(coordinates.begin(), coordinates.end(), coordinateCmp);
        //vector<int*> coordinates1 = combinecoord_for_realaccount(coordinates);//白纸上数字,实际进账单帐号
        //vector<int*> coordinates1 = combinecoord_for_five(coordinates, contours);//白纸上数字,实际进账单帐号
        vector<int*> coordinates1 = combinecoord_for_five_h(coordinates, contours);//白纸上数字,实际进账单帐号
        //vector<int*> coordinates1 = combinecoord1(coordinates, true);//实际票据上小写数字
        vector<int> wid;
        vector<Mat> tempMat;
        for (int i = 0; i < coordinates1.size(); i++) {
//            int x1 = coordinates1[i][0];
//            int x2 = coordinates1[i][1];
//            int y1 = coordinates1[i][2];
//            int y2 = coordinates1[i][3];
//            Mat temp = img1clone(Range(y1, y2 + 1), Range(x1, x2 + 1));
//            Mat tempclone1 = temp.clone();
            vector<Point> finalcontour;
            for (int i1 = 0; i1 < contours[coordinates1[i][4]].size(); i1++) {
                finalcontour.push_back(contours[coordinates1[i][4]][i1]);
            }
            if (coordinates1[i][5] != -1) {
                for (int i2 = 0; i2 < contours[coordinates1[i][5]].size(); i2++) {
                    finalcontour.push_back(contours[coordinates1[i][5]][i2]);
                }
            }
            sort(finalcontour.begin(), finalcontour.end(), yCmp);
            int yup = finalcontour[0].y;
            int ydown = finalcontour[finalcontour.size() - 1].y;
            int y1 = yup;
            int y2 = ydown;
            sort(finalcontour.begin(), finalcontour.end(), xCmp);
            int x1 = finalcontour[0].x;
            int x2 = finalcontour[finalcontour.size() - 1].x;
            Mat temp = img1clone(Range(y1, y2 + 1), Range(x1, x2 + 1));
            Mat tempclone1 = temp.clone();
            vector<vector<int>> filled = contoursRows(finalcontour, yup, ydown);
            for (int j = 0; j < tempclone1.rows; j++) {
                for (int l = 0; l < tempclone1.cols; l++) {
                    if (j < filled.size()){
                    if (filled[j].size() == 1) {
                        int x = filled[j][0];
                        if (l != (x - x1)) {
                            tempclone1.at<uchar>(j, l) = 0;
                        }
                    }
                    else if (filled[j].size() == 0)
                    {
                        tempclone1.at<uchar>(j, l) = 0;
                    }
                    else
                    {
                        int xl = filled[j][0];
                        int xr = filled[j][1];
                        if ((l < (xl - x1)) || (l >(xr - x1))) {
                            tempclone1.at<uchar>(j, l) = 0;
                        }
                    }
                }
                }
            }
            wid.push_back(tempclone1.cols);
            tempMat.push_back(tempclone1);
        }
        for (int fsp = 0; fsp < tempMat.size(); fsp++) {
            Mat matfsp = tempMat[fsp];
            Mat a1res = cut_margin_for_pic(matfsp);
            bitwise_not(a1res, a1res);
            if (a1res.cols != 0){
            v.push_back(a1res);}
        }
//    }
        if (debug){
            if (false){
                for(int i=0;i<v.size();i++){
                    char save_file_name[300];
                    string s="/home/ubuntu/桌面/tempresult1/" + filename;
                    QString s1 = QString::fromStdString(s);
                    QDir dir;
                    if (!dir.exists(s1)){
                        bool res = dir.mkpath(s1);
                    }
                    string s2=s+"/%d.png";
                    sprintf(save_file_name,s2.c_str(), i);
                    imwrite(save_file_name,v[i]);
                }
            }
            else{
                for (int i = 0; i < coordinates1.size(); i++){
                    rectangle(img1_cutline, Point(coordinates1[i][0], coordinates1[i][2]),
                                        Point(coordinates1[i][1], coordinates1[i][3]), Scalar(0), 0.2);
                }
                char save_file_name[300];
                string s="/home/ubuntu/桌面/tempresult1/" + filename;
                QString s1 = QString::fromStdString(s);
                QDir dir;
                if (!dir.exists(s1)){
                    bool res = dir.mkpath(s1);
                }
                string s2=s+"/%d.png";
                sprintf(save_file_name,s2.c_str(), 0);
                imwrite(save_file_name,img1_cutline);
            }
        }
    return v;
}

std::vector<cv::Mat> cutHWLowerNumByH3(Mat img1) {
    vector<Mat> v;
    //string ff = fileName.substr(fileName.length() - 4, fileName.length());
    //if (ff.compare(".png") == 0) {

        bitwise_not(img1, img1);
        //imshow("qsddwawg", img1);
        //deleteLines2(img1, 30);
        //imshow("qsddwg", img1);
        //RemoveSmallRegion(img1, img1, 20, 1, 1);
        //imshow("contomg", img1);
        Mat img11 = img1.clone();
        vector<vector<Point>> contours1;
        Mat img1clone = img11.clone();
        findContours(img11,
                     contours1,
                     CV_RETR_EXTERNAL,
                     //CV_RETR_TREE,
                     CV_CHAIN_APPROX_NONE);
        /*Mat contoursImg = Mat(img1.rows, img1.cols, CV_8U);
            drawContours(contoursImg, contours1);
            imshow("contoursImg", contoursImg);
            waitKey();*/
        vector<vector<Point>> contours = removeShortContoursfornumber(contours1, img11.rows);
        vector<int*> coordinates = CCPoints2Coordinates1(contours);
        sort(coordinates.begin(), coordinates.end(), coordinateCmp);
        vector<int*> coordinates1 = combinecoord1(coordinates, true);

        vector<int> wid;
        vector<int> hei;
        vector<int> contourwid;
        //vector<int> area;
        vector<Mat> tempMat;
        for (int i = 0; i < coordinates1.size(); i++) {
            int x1 = coordinates1[i][0];
            int x2 = coordinates1[i][1];
            int y1 = coordinates1[i][2];
            int y2 = coordinates1[i][3];
            Mat temp = img1clone(Range(y1, y2 + 1), Range(x1, x2 + 1));

            Mat tempclone = temp.clone();
            Mat tempclone1 = temp.clone();
            vector<Point> finalcontour;
            for (int i1 = 0; i1 < contours[coordinates1[i][4]].size(); i1++) {
                finalcontour.push_back(contours[coordinates1[i][4]][i1]);
            }
            if (coordinates1[i][5] != -1) {
                for (int i2 = 0; i2 < contours[coordinates1[i][5]].size(); i2++) {
                    finalcontour.push_back(contours[coordinates1[i][5]][i2]);
                }
            }
            sort(finalcontour.begin(), finalcontour.end(), yCmp);
            int yup = finalcontour[0].y;
            int ydown = finalcontour[finalcontour.size() - 1].y;
            vector<vector<int>> filled = contoursRows(finalcontour, yup, ydown);
            for (int j = 0; j < tempclone1.rows; j++) {
                for (int l = 0; l < tempclone1.cols; l++) {
                    if (filled[j].size() == 1) {
                        int x = filled[j][0];
                        if (l != (x - x1)) {
                            tempclone1.at<uchar>(j, l) = 0;
                        }
                    }
                    else if (filled[j].size() == 0)
                    {
                        tempclone1.at<uchar>(j, l) = 0;
                    }
                    else
                    {
                        int xl = filled[j][0];
                        int xr = filled[j][1];
                        if ((l < (xl - x1)) || (l >(xr - x1))) {
                            tempclone1.at<uchar>(j, l) = 0;
                        }
                    }
                }
            }
            /*imshow("nh", tempclone1);
                waitKey();*/
            wid.push_back(tempclone1.cols);
            hei.push_back(tempclone1.rows);
            tempMat.push_back(tempclone1);
            contourwid.push_back(finalcontour.size());
            //vector<vector<Point>> contours2;
            //Mat recontour = tempclone1.clone();
            //findContours(recontour,
            //	contours2,
            //	//CV_RETR_EXTERNAL,
            //	CV_RETR_TREE,
            //	CV_CHAIN_APPROX_NONE);
            //vector<Point> recons;
            //for (int i = 0; i < contours2.size(); i++)
            //	for (int j = 0; j < contours2[i].size(); j++)
            //		recons.push_back(contours2[i][j]);
            //area.push_back(contourArea(recons));
        }


        int midwid = Middle(wid, wid.size());
        int midhei = Middle(hei, hei.size());
        int midconwid = Middle(contourwid, contourwid.size());
        //int midarea = Middle(area, area.size());
        //sort(contourwid.begin(), contourwid.end(), widthCmp);
        //sort(wid.begin(), wid.end(), widthCmp);
        //sort(area.begin(), area.end(), widthCmp);
        int numpic = 0;
        for (int fsp = 0; fsp < tempMat.size(); fsp++) {
            Mat matfsp = tempMat[fsp];
            int cwid = matfsp.cols;
            int chei = matfsp.rows;
            int conwid = contourwid[fsp];
            float cnum = conwid * 1.0 / midconwid * 1.0;
            float colnum = cwid * 1.0 / midwid * 1.0;
            float heirotoi = chei * 1.0 / midhei;
            if (true){		//格子里的数字用不黏连切
                IplImage t1= IplImage(matfsp);
                IplImage *a1ipl = &t1;
                int x1 = getcutpos(a1ipl, 0)[0];
                int x2 = getcutpos(a1ipl, 0)[1];
                int y1 = getcutpos(a1ipl, 0)[2];
                int y2 = getcutpos(a1ipl, 0)[3];
                Mat a1res = matfsp(Range(y1, y2), Range(x1, x2));
                bitwise_not(a1res, a1res);
                if (a1res.cols != 0){
                v.push_back(a1res);}
                char save_file_name[300];
                string s1="/home/ubuntu/testinBank/tempresults/cols/";
                string s2=s1+"%d.png";
                sprintf(save_file_name,s2.c_str(), numpic);
                imwrite(save_file_name,a1res);
                numpic++;
            }

        }
    //}
    return v;
}

std::vector<cv::Mat> cutHWLowerNumByH4(Mat img1, int num_of_adh, int add_weishu, string filename, bool debug) {
    vector<Mat> v;
    Mat img1_cutline = img1.clone();
            bitwise_not(img1, img1);
            Mat j = img1.clone();
            vector<vector<Point>> contours1;
            findContours(img1,
                         contours1,
                         CV_RETR_EXTERNAL,
                         //CV_RETR_TREE,
                         CV_CHAIN_APPROX_NONE);
            /*Mat contoursImg = Mat(img1.rows, img1.cols, CV_8U);
                drawContours(contoursImg, contours1);
                imshow("contoursImg", contoursImg);
                waitKey();*/
            vector<vector<Point>> contours = removeShortContoursfornumber(contours1, img1.rows);
            vector<int*> coordinates = CCPoints2Coordinates1(contours);
            sort(coordinates.begin(), coordinates.end(), coordinateCmp);
            vector<int*> coordinates1 = combinecoord_for_realaccount(coordinates);


            if (add_weishu == 0){
                coordinates1.erase(coordinates1.begin());
                coordinates1.erase(coordinates1.begin());
            }
            else if (add_weishu == 1/* || add_weishu == 3*/){
                coordinates1.erase(coordinates1.begin());
            }


            vector<int> wid;
            vector<int> hei;
            vector<int> contourwid;
            vector<Mat> tempMat;
            for (int i = 0; i < coordinates1.size(); i++) {
                vector<Point> finalcontour;
                for (int i1 = 0; i1 < contours[coordinates1[i][4]].size(); i1++) {
                    finalcontour.push_back(contours[coordinates1[i][4]][i1]);
                }
                if (coordinates1[i][5] != -1) {
                    for (int i2 = 0; i2 < contours[coordinates1[i][5]].size(); i2++) {
                        finalcontour.push_back(contours[coordinates1[i][5]][i2]);
                    }
                }
                sort(finalcontour.begin(), finalcontour.end(), yCmp);
                int yup = finalcontour[0].y;
                int ydown = finalcontour[finalcontour.size() - 1].y;
                int y1 = yup;
                int y2 = ydown;
                sort(finalcontour.begin(), finalcontour.end(), xCmp);
                int x1 = finalcontour[0].x;
                int x2 = finalcontour[finalcontour.size() - 1].x;
                Mat temp = j(Range(y1, y2 + 1), Range(x1, x2 + 1));
                Mat tempclone1 = temp.clone();
                vector<vector<int>> filled = contoursRows(finalcontour, yup, ydown);
                for (int j = 0; j < tempclone1.rows; j++) {
                    for (int l = 0; l < tempclone1.cols; l++) {
                        if (filled[j].size() == 1) {
                            int x = filled[j][0];
                            if (l != (x - x1)) {
                                tempclone1.at<uchar>(j, l) = 0;
                            }
                        }
                        else if (filled[j].size() == 0)
                        {
                            tempclone1.at<uchar>(j, l) = 0;
                        }
                        else
                        {
                            int xl = filled[j][0];
                            int xr = filled[j][1];
                            if ((l < (xl - x1)) || (l >(xr - x1))) {
                                tempclone1.at<uchar>(j, l) = 0;
                            }
                        }
                    }
                }
                /*imshow("nh", tempclone1);
                    waitKey();*/
                wid.push_back(tempclone1.cols);
                hei.push_back(tempclone1.rows);
                tempMat.push_back(tempclone1);
                contourwid.push_back(finalcontour.size());
            }


            int midwid = Middle(wid, wid.size());
            int midconwid = Middle(contourwid, contourwid.size());
            int numpic = 0;
            vector<int> adh_id;
            vector<int*> adh1;
            vector<int*> adh2;
            vector<int*> adh3;
            vector<int*> adh4;
            vector<int*> adh5;
            for (int fsp = 0; fsp < tempMat.size(); fsp++) {
                Mat matfsp = tempMat[fsp];
                int cwid = matfsp.cols;
                int conwid = contourwid[fsp];
                float cnum = conwid * 1.0 / midconwid * 1.0;
                float colnum = cwid * 1.0 / midwid * 1.0;
                if (colnum > 1.7){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh1.push_back(a);
                }
                if (colnum > 1.5 && colnum <= 1.7 && cnum > 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh2.push_back(a);
                }
                if (colnum <= 1.5 && cnum > 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = cnum;
                    adh3.push_back(a);
                }
                if (colnum > 1.5 && colnum <= 1.7 && cnum <= 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh4.push_back(a);
                }
                if (colnum <= 1.5 && cnum <= 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh5.push_back(a);
                }
            }
            if (adh1.size() > 0){
                sort(adh1.begin(), adh1.end(), contoursizeCmp);
                for (int i = adh1.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh1[i][0]);
                }
            }
            if (adh2.size() > 0){
                sort(adh2.begin(), adh2.end(), contoursizeCmp);
                for (int i = adh2.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh2[i][0]);
                }
            }
            if (adh3.size() > 0){
                sort(adh3.begin(), adh3.end(), contoursizeCmp);
                for (int i = adh3.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh3[i][0]);
                }
            }
            if (adh4.size() > 0){
                sort(adh4.begin(), adh4.end(), contoursizeCmp);
                for (int i = adh4.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh4[i][0]);
                }
            }
            if (adh5.size() > 0){
                sort(adh5.begin(), adh5.end(), contoursizeCmp);
                for (int i = adh5.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh5[i][0]);
                }
            }
            int count_adh = 0;
            map<int, vector<Mat>> cuted_adh;
            for (int i = 0; i < adh_id.size();i++){
                Mat test = tempMat[adh_id[i]];
                Mat test_clone = test.clone();
                Mat five_clone = test.clone();
                vector<vector<Point>> contoursfive;
                findContours(five_clone,contoursfive,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
                removeShortContours(contoursfive, 3, 500);
                if (contoursfive.size() > 1){
                    continue;
                }

                int *range = new int[2];
                int cp = MapsumtoCut(test_clone, range);
                if (cp == 0){
                    bool false_cut = true;
                    vector<Mat> cuted_mat;
                    vector<Mat> split_imgs = get_split_imgs_with_skeleton(test, skeleton_debug);
                    for (int split_id = 0; split_id < split_imgs.size(); split_id++){
                        Mat split_cutmargin;
                        split_cutmargin = cut_margin_for_pic(split_imgs[split_id]);
                        if (split_cutmargin.rows < min_length_for_num || split_cutmargin.cols < min_length_for_num){
                            false_cut = false;
                            break;
                        }
                        cuted_mat.push_back(split_cutmargin);
                    }
                    if (false_cut){
                        cuted_adh[adh_id[i]] = cuted_mat;
                        count_adh++;
                    }
                    else{
                        continue;
                    }
                }
                else{
                    Mat mat1 = test(Range(0, test.rows), Range(0, range[0]));
                    Mat mat2 = test(Range(0, test.rows), Range(range[1], test.cols));
                    Mat a1res1 = cut_margin_for_pic(mat1);
                    Mat a1res2 = cut_margin_for_pic(mat2);
                    if (a1res1.rows < min_length_for_num || a1res1.cols < min_length_for_num
                            || a1res2.rows < min_length_for_num || a1res2.cols < min_length_for_num){
                        continue;
                    }
                    else{
                        vector<Mat> cuted_mat;
                        cuted_mat.push_back(mat1);
                        cuted_mat.push_back(mat2);
                        cuted_adh[adh_id[i]] = cuted_mat;
                        count_adh++;
                    }
                }
                if (count_adh == num_of_adh){
                    break;
                }
            }

            for (int fsp = 0; fsp < tempMat.size(); fsp++) {
                Mat matfsp = tempMat[fsp];
                if (false){		//格子里的数字用不黏连切
                    cout<<"Hahaha, what a shit project！！！"<<endl;
                }
                else {
                    map<int, vector<Mat>>::iterator find;
                    find = cuted_adh.find(fsp);
                    if (find != cuted_adh.end()){
                        vector<Mat> mat = find->second;
                        for (int i = 0; i < mat.size(); i++){
                            bitwise_not(mat[i], mat[i]);
                            v.push_back(mat[i]);
                            if (debug){
                            char save_file_name[300];
                            string s="/home/ubuntu/桌面/tempresult/" + filename;
                            QString s1 = QString::fromStdString(s);
                            QDir dir;
                            if (!dir.exists(s1)){
                                bool res = dir.mkpath(s1);
                            }
                            string s2=s+"/%d.png";
                            sprintf(save_file_name,s2.c_str(), numpic);
                            imwrite(save_file_name,mat[i]);
                            numpic++;}
                        }
                    }
                    else{
                        Mat cut_margin = cut_margin_for_pic(matfsp);
                        bitwise_not(cut_margin, cut_margin);
                        v.push_back(cut_margin);
                        if (debug){
                        char save_file_name[300];
                        string s="/home/ubuntu/桌面/tempresult/" + filename;
                        QString s1 = QString::fromStdString(s);
                        QDir dir;
                        if (!dir.exists(s1)){
                            bool res = dir.mkpath(s1);
                        }
                        string s2=s+"/%d.png";
                        sprintf(save_file_name,s2.c_str(), numpic);
                        imwrite(save_file_name,cut_margin);
                        numpic++;}
                    }
                }
            }

            if (debug){
                for (int i = 0; i < coordinates1.size(); i++){
                    rectangle(img1_cutline, Point(coordinates1[i][0], coordinates1[i][2]),
                                        Point(coordinates1[i][1], coordinates1[i][3]), Scalar(0), 0.2);
                }
                char save_file_name[300];
                string s="/home/ubuntu/桌面/tempresult/" + filename;
                QString s1 = QString::fromStdString(s);
                QDir dir;
                if (!dir.exists(s1)){
                    bool res = dir.mkpath(s1);
                }
                string s2=s+"/%d.png";
                sprintf(save_file_name,s2.c_str(), 8888);
                imwrite(save_file_name,img1_cutline);
            }
        return v;

}

std::vector<cv::Mat> cutHWLowerNumByH5(Mat img1, int num_of_adh, bool debug, string filename) {
    vector<Mat> v;
    Mat img1_cutline = img1.clone();
            bitwise_not(img1, img1);
            Mat j = img1.clone();
            vector<vector<Point>> contours1;
            findContours(img1,
                         contours1,
                         CV_RETR_EXTERNAL,
                         //CV_RETR_TREE,
                         CV_CHAIN_APPROX_NONE);
            /*Mat contoursImg = Mat(img1.rows, img1.cols, CV_8U);
                drawContours(contoursImg, contours1);
                imshow("contoursImg", contoursImg);
                waitKey();*/
            vector<vector<Point>> contours = removeShortContoursfornumber(contours1, img1.rows);
            vector<int*> coordinates = CCPoints2Coordinates1(contours);
            sort(coordinates.begin(), coordinates.end(), coordinateCmp);
            //vector<int*> coordinates1 = combinecoord_for_realaccount(coordinates);
            //vector<int*> coordinates1 = combinecoord_for_five(coordinates, contours);
            vector<int*> coordinates1 = combinecoord_for_five_h(coordinates, contours);



            vector<int> wid;
            vector<int> hei;
            vector<int> contourwid;
            vector<Mat> tempMat;
            for (int i = 0; i < coordinates1.size(); i++) {
                vector<Point> finalcontour;
                for (int i1 = 0; i1 < contours[coordinates1[i][4]].size(); i1++) {
                    finalcontour.push_back(contours[coordinates1[i][4]][i1]);
                }
                if (coordinates1[i][5] != -1) {
                    for (int i2 = 0; i2 < contours[coordinates1[i][5]].size(); i2++) {
                        finalcontour.push_back(contours[coordinates1[i][5]][i2]);
                    }
                }
                sort(finalcontour.begin(), finalcontour.end(), yCmp);
                int yup = finalcontour[0].y;
                int ydown = finalcontour[finalcontour.size() - 1].y;
                int y1 = yup;
                int y2 = ydown;
                sort(finalcontour.begin(), finalcontour.end(), xCmp);
                int x1 = finalcontour[0].x;
                int x2 = finalcontour[finalcontour.size() - 1].x;
                Mat temp = j(Range(y1, y2 + 1), Range(x1, x2 + 1));
                Mat tempclone1 = temp.clone();
                vector<vector<int>> filled = contoursRows(finalcontour, yup, ydown);
                for (int j = 0; j < tempclone1.rows; j++) {
                    for (int l = 0; l < tempclone1.cols; l++) {
                        if (filled[j].size() == 1) {
                            int x = filled[j][0];
                            if (l != (x - x1)) {
                                tempclone1.at<uchar>(j, l) = 0;
                            }
                        }
                        else if (filled[j].size() == 0)
                        {
                            tempclone1.at<uchar>(j, l) = 0;
                        }
                        else
                        {
                            int xl = filled[j][0];
                            int xr = filled[j][1];
                            if ((l < (xl - x1)) || (l >(xr - x1))) {
                                tempclone1.at<uchar>(j, l) = 0;
                            }
                        }
                    }
                }
                /*imshow("nh", tempclone1);
                    waitKey();*/
                wid.push_back(tempclone1.cols);
                hei.push_back(tempclone1.rows);
                tempMat.push_back(tempclone1);
                contourwid.push_back(finalcontour.size());
            }


            int midwid = Middle(wid, wid.size());
            int midconwid = Middle(contourwid, contourwid.size());
            int numpic = 0;
            vector<int> adh_id;
            vector<int*> adh1;
            vector<int*> adh2;
            vector<int*> adh3;
            vector<int*> adh4;
            vector<int*> adh5;
            for (int fsp = 0; fsp < tempMat.size(); fsp++) {
                Mat matfsp = tempMat[fsp];
                int cwid = matfsp.cols;
                int conwid = contourwid[fsp];
                float cnum = conwid * 1.0 / midconwid * 1.0;
                float colnum = cwid * 1.0 / midwid * 1.0;
                if (colnum > 1.7){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh1.push_back(a);
                }
                if (colnum > 1.5 && colnum <= 1.7 && cnum > 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh2.push_back(a);
                }
                if (colnum <= 1.5 && cnum > 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = cnum;
                    adh3.push_back(a);
                }
                if (colnum > 1.5 && colnum <= 1.7 && cnum <= 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh4.push_back(a);
                }
                if (colnum <= 1.5 && cnum <= 1.5){
                    int* a = new int[2];
                    a[0] = fsp;
                    a[1] = colnum;
                    adh5.push_back(a);
                }
            }
            if (adh1.size() > 0){
                sort(adh1.begin(), adh1.end(), contoursizeCmp);
                for (int i = adh1.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh1[i][0]);
                }
            }
            if (adh2.size() > 0){
                sort(adh2.begin(), adh2.end(), contoursizeCmp);
                for (int i = adh2.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh2[i][0]);
                }
            }
            if (adh3.size() > 0){
                sort(adh3.begin(), adh3.end(), contoursizeCmp);
                for (int i = adh3.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh3[i][0]);
                }
            }
            if (adh4.size() > 0){
                sort(adh4.begin(), adh4.end(), contoursizeCmp);
                for (int i = adh4.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh4[i][0]);
                }
            }
            if (adh5.size() > 0){
                sort(adh5.begin(), adh5.end(), contoursizeCmp);
                for (int i = adh5.size() - 1; i >= 0; i--){
                    adh_id.push_back(adh5[i][0]);
                }
            }
            int count_adh = 0;
            map<int, vector<Mat>> cuted_adh;
            for (int i = 0; i < adh_id.size();i++){
                Mat test = tempMat[adh_id[i]];
                Mat test_clone = test.clone();
                Mat five_clone = test.clone();
                vector<vector<Point>> contoursfive;
                findContours(five_clone,contoursfive,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
                removeShortContours(contoursfive, 3, 500);
                if (contoursfive.size() > 1){
                    Mat five_line;
                    test_clone = erase_for_five(test_clone, five_line);
                }

                int *range = new int[2];
                int cp = MapsumtoCut(test_clone, range);
                if (cp == 0){
                    bool false_cut = true;
                    vector<Mat> cuted_mat;
                    vector<Mat> split_imgs = get_split_imgs_with_skeleton(test, skeleton_debug);
                    for (int split_id = 0; split_id < split_imgs.size(); split_id++){
                        Mat split_cutmargin;
                        split_cutmargin = cut_margin_for_pic(split_imgs[split_id]);
                        if (split_cutmargin.rows < min_length_for_num || split_cutmargin.cols < min_length_for_num){
                            false_cut = false;
                            break;
                        }
                        cuted_mat.push_back(split_cutmargin);
                    }
                    if (false_cut){
                        cuted_adh[adh_id[i]] = cuted_mat;
                        count_adh++;
                    }
                    else{
                        continue;
                    }
                }
                else{
                    Mat mat1 = test(Range(0, test.rows), Range(0, range[0]));
                    Mat mat2 = test(Range(0, test.rows), Range(range[1], test.cols));
                    Mat a1res1 = cut_margin_for_pic(mat1);
                    Mat a1res2 = cut_margin_for_pic(mat2);
                    if (a1res1.rows < min_length_for_num || a1res1.cols < min_length_for_num
                            || a1res2.rows < min_length_for_num || a1res2.cols < min_length_for_num){
                        continue;
                    }
                    else{
                        vector<Mat> cuted_mat;
                        cuted_mat.push_back(mat1);
                        cuted_mat.push_back(mat2);
                        cuted_adh[adh_id[i]] = cuted_mat;
                        count_adh++;
                    }
                }
                if (count_adh == num_of_adh){
                    break;
                }
            }

            for (int fsp = 0; fsp < tempMat.size(); fsp++) {
                Mat matfsp = tempMat[fsp];
                if (false){		//格子里的数字用不黏连切
                    cout<<"Hahaha, what a shit project！！！"<<endl;
                }
                else {
                    map<int, vector<Mat>>::iterator find;
                    find = cuted_adh.find(fsp);
                    if (find != cuted_adh.end()){
                        vector<Mat> mat = find->second;
                        for (int i = 0; i < mat.size(); i++){
                            bitwise_not(mat[i], mat[i]);
                            v.push_back(mat[i]);
                            if (debug){
                            char save_file_name[300];
                            string s="/home/ubuntu/桌面/tempresult/" + filename;
                            QString s1 = QString::fromStdString(s);
                            QDir dir;
                            if (!dir.exists(s1)){
                                bool res = dir.mkpath(s1);
                            }
                            string s2=s+"/%d.png";
                            sprintf(save_file_name,s2.c_str(), numpic);
                            imwrite(save_file_name,mat[i]);
                            numpic++;}
                        }
                    }
                    else{
                        Mat cut_margin = cut_margin_for_pic(matfsp);
                        bitwise_not(cut_margin, cut_margin);
                        v.push_back(cut_margin);
                        if (debug){
                        char save_file_name[300];
                        string s="/home/ubuntu/桌面/tempresult/" + filename;
                        QString s1 = QString::fromStdString(s);
                        QDir dir;
                        if (!dir.exists(s1)){
                            bool res = dir.mkpath(s1);
                        }
                        string s2=s+"/%d.png";
                        sprintf(save_file_name,s2.c_str(), numpic);
                        imwrite(save_file_name,cut_margin);
                        numpic++;}
                    }
                }
            }

            if (debug){
//                for (int i = 0; i < coordinates1.size(); i++){
//                    rectangle(img1_cutline, Point(coordinates1[i][0], coordinates1[i][2]),
//                                        Point(coordinates1[i][1], coordinates1[i][3]), Scalar(0), 0.2);
//                }
//                char save_file_name[300];
//                string s="/home/ubuntu/桌面/tempresult/" + filename;
//                QString s1 = QString::fromStdString(s);
//                QDir dir;
//                if (!dir.exists(s1)){
//                    bool res = dir.mkpath(s1);
//                }
//                string s2=s+"/%d.png";
//                sprintf(save_file_name,s2.c_str(), 8888);
//                imwrite(save_file_name,img1_cutline);
            }
        return v;

}


vector<int*> getCoord_by_tessact(Mat gray, tesseract::TessBaseAPI *api) {
    vector<int*> coord_inint;
    vector<int*> coord_result;

    api->SetImage((uchar*)gray.data, gray.cols, gray.rows, 1, gray.cols);
    Boxa* boxes = api->GetComponentImages(tesseract::RIL_SYMBOL, true, false, 0, NULL, NULL, NULL);
    //printf("Found %d textline image components.\n", boxes->n);
    int num = 0;
    if(!boxes){
        try{num = boxes->n;
        }catch(exception &e){
            num=0;
        }
    }
    else{
        num = boxes->n;
    }
    for (int i = 0; i < num; i++) {
        BOX* box = boxaGetBox(boxes, i, L_CLONE);
        api->SetRectangle(box->x, box->y, box->w, box->h);
        if (box->w >= 2 && box->h >= 3) {		//筛掉细线
            if ((box->w <= 3 && box->h <= gray.rows * 0.9) || box->w > 3) {
                int* coord = new int[4];
                coord[0] = box->x;
                coord[1] = box->x + box->w;
                coord[2] = box->y;
                coord[3] = box->y + box->h;
                coord_inint.push_back(coord);
            }
        }
    }
    sort(coord_inint.begin(), coord_inint.end(), coordinateCmp);
    coord_result.push_back(coord_inint[0]);
    for (int i = 1; i < coord_inint.size(); i++) {
        int dis = coord_result[coord_result.size() - 1][1] - coord_inint[i][0];
        if (dis > -3) {		//合并阈值-3
            int* coord = new int[4];
            coord[0] = coord_result[coord_result.size() - 1][0];
            coord[1] = coord_inint[i][1];
            coord[2] = Min(coord_result[coord_result.size() - 1][2], coord_inint[i][2]);
            coord[3] = Max(coord_result[coord_result.size() - 1][3], coord_inint[i][3]);
            coord_result.pop_back();
            coord_result.push_back(coord);
        }
        else {
            int wid_left = coord_result[coord_result.size() - 1][1] - coord_result[coord_result.size() - 1][0];
            int wid_right = coord_inint[i][1] - coord_inint[i][0];
            if (dis > -5 && wid_left < 6 && wid_right < 6) {	//切碎的两半0，在上面合并时没有满足阈值合并，此时再合并
                int* coord = new int[4];
                coord[0] = coord_result[coord_result.size() - 1][0];
                coord[1] = coord_inint[i][1];
                coord[2] = Min(coord_result[coord_result.size() - 1][2], coord_inint[i][2]);
                coord[3] = Max(coord_result[coord_result.size() - 1][3], coord_inint[i][3]);
                coord_result.pop_back();
                coord_result.push_back(coord);
            }
            else {
                coord_result.push_back(coord_inint[i]);
            }
        }


//        for (int i = 0; i < coord_result.size(); i++) {
//            line(gray, Point(coord_result[i][0], coord_result[i][2]), Point(coord_result[i][0], coord_result[i][3]), Scalar(0, 0, 255), 1, 4);
//            line(gray, Point(coord_result[i][0], coord_result[i][2]), Point(coord_result[i][1], coord_result[i][2]), Scalar(0, 0, 255), 1, 4);
//            line(gray, Point(coord_result[i][1], coord_result[i][2]), Point(coord_result[i][1], coord_result[i][3]), Scalar(0, 0, 255), 1, 4);
//            line(gray, Point(coord_result[i][0], coord_result[i][3]), Point(coord_result[i][1], coord_result[i][3]), Scalar(0, 0, 255), 1, 4);
//        }
//        imshow("bfg", gray);
//        waitKey();
    }
    free(boxes);
    return coord_result;
}


vector<Mat> get_splitMat(Mat temp) {
    vector<Mat> result;
    vector<Mat> channels;
    split(temp, channels);
    Mat red = channels[2];
    Mat red_clone = red.clone();
    IplImage Ipltemp = IplImage(red_clone);
    IplImage *ip = &Ipltemp;
    int thres = otsu2(ip) + 10;
    //threshold(red, red, 0, 255, THRESH_OTSU);
    threshold(red, red, thres, 255, THRESH_BINARY);
    bitwise_not(red, red);

    cvtColor(temp, temp, CV_BGR2GRAY);
    threshold(temp, temp, 0, 255, THRESH_OTSU);
    bitwise_not(temp, temp);

    //int* check_map = new int[temp.rows];
    int* check_map_loc = new int[temp.cols];
    //map_to_remove(temp, check_map);
    map_to_locate(temp, check_map_loc);
    vector<int> split_coord_before;
    vector<int> split_coord;
    /*for (int i = 0; i < temp.rows; i++) {
        if (check_map[i] == temp.cols) {
            line(temp, Point(0, i), Point(temp.cols - 1, i), Scalar(0), 2, 4);
        }
    }*/
    for (int i = 0; i < temp.cols; i++) {
        if (check_map_loc[i] == temp.rows) {
            if (temp.cols > 12) {		//太细的是数字，不会有红线
                line(red, Point(i, 0), Point(i, temp.rows - 1), Scalar(0), 1, 4);
            }
            split_coord_before.push_back(i);
        }
    }
    split_coord_before.insert(split_coord_before.begin(), 0);
    split_coord_before.push_back(temp.cols);
    if (split_coord_before.size() != 0 && temp.cols > 30) {		//对那些宽的有红线的切
        for (int i = 1; i < split_coord_before.size(); i++) {
            int pre = split_coord_before[i - 1];
            int cur = split_coord_before[i];
            if ((cur - pre) < 5) {		//相同红线距离最短值
                continue;
            }
            else {
                split_coord.push_back(cur);
            }
        }
        split_coord.insert(split_coord.begin(), 0);
    }
    if (split_coord.size() > 0 && temp.cols > 30) {
        for (int i = 0; i < split_coord.size() - 1; i++) {
            Mat block = red(Range(0, temp.rows), Range(split_coord[i], split_coord[i + 1]));
            result.push_back(block);
        }
    }
    else {
        result.push_back(red);
    }
    /*imshow("g", red);
    waitKey();*/
    return result;
}


void removeShortContours(vector<vector<Point>>& contours) {
    vector<vector<Point>>::iterator itc = contours.begin();
    while (itc != contours.end())
    {
        vector<Point> con = *itc;
        sort(con.begin(), con.end(), yCmp);
        int conSize = con.size();
        int count2 = 0;
        int norm2 = con[conSize - 1].y;
        int count1 = 0;
        int norm1 = con[0].y;
        for (int i = 0; i < conSize; i++) {
            if (norm2 == con[i].y)
                count2++;
            if (norm1 == con[i].y)
                count1++;
        }
        int count = 1;
        for (int i = 1; i < conSize; i++) {
            if (con[i].y > con[i - 1].y) {
                count++;
            }
        }
        if ((count2 < (conSize * 0.9) && count1 < (conSize * 0.9)) /*&& count > 2*/)
            itc = contours.erase(itc);
        else
        {
            ++itc;
        }
    }
}

Mat CutNoise_for_split_downnum(Mat img) {
    vector<vector<Point>> contours;
    for (int k = 0; k <= img.cols - 1; k++) {
        img.at<uchar>(0, k) = 0;
    }
    /*for (int k = 0; k <= img.cols - 1; k++) {
        img.at<uchar>(img.rows - 1, k) = 0;
    }*/
    Mat img1clone = img.clone();
    findContours(img,
                 contours,
                 CV_RETR_EXTERNAL,
                 //CV_RETR_TREE,
                 CV_CHAIN_APPROX_NONE);
    removeShortContours(contours);

    vector<int*> coordinates = CCPoints2Coordinates(contours);
    for (int i = 0; i < coordinates.size(); i++) {
        int x1 = coordinates[i][0];
        vector<Point> finalcontour = contours[i];
        sort(finalcontour.begin(), finalcontour.end(), yCmp);
        int yup = finalcontour[0].y;
        int ydown = finalcontour[finalcontour.size() - 1].y;
        vector<vector<int>> filled = contoursRows(finalcontour, yup, ydown);
        int count = 0;
        for (int j = yup; j <= ydown; j++) {
            if (filled[count].size() == 1) {
                img1clone.at<uchar>(j, filled[count][0]) = 0;
            }
            else if (filled[count].size() == 2) {
                int xl = filled[count][0];
                int xr = filled[count][1];
                for (int k = xl; k <= xr; k++) {
                    img1clone.at<uchar>(j, k) = 0;
                }
            }
            else {
                for (int k = 0; k <= img1clone.cols - 1; k++) {
                    img1clone.at<uchar>(j, k) = 0;
                }
            }
            count++;
        }
    }
    return img1clone;
}
