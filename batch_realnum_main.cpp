#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <fstream>
#include <iomanip>

#include <math.h>
#include <string.h>
#include <stdio.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <QString>
#include <QDir>
#include <QFile>
#include <QDebug>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <tesseract/strngs.h>

#include "cutcol.h"
#include "caffe_inference.h"
#include "time.h"
#include "segmentpic.h"

using namespace cv;
using namespace std;

string getNum11(vector<Mat> imgs, Classifier caffe_model){
    string result = "";
    Mat final_img;
    int count = 0;
    for (int i = 0; i < imgs.size(); i++){
        int height = imgs[i].rows;
        int width = imgs[i].cols;
        //image inverse
        bitwise_not(imgs[i], final_img);

        //image square
        if(width > height) {
            int gap = width - height;
            int add_pix = gap / 2;
            cv::copyMakeBorder(final_img, final_img, add_pix, add_pix, 0, 0, cv::BORDER_CONSTANT,cv::Scalar(0));
        }else {
            int gap = height - width;
            int add_pix = gap / 2;
            cv::copyMakeBorder(final_img, final_img, 0, 0, add_pix, add_pix, cv::BORDER_CONSTANT,cv::Scalar(0));
        }

//        char save_name[300];
//        string s1 = "/home/ubuntu/桌面/11111";
//        string s= s1 + "/%d.png";
//        sprintf(save_name, s.c_str(), count);
//        imwrite(save_name, final_img);
//        count++;
        cv::copyMakeBorder(final_img, final_img, 2, 2, 2, 2, cv::BORDER_CONSTANT,cv::Scalar(0));
        vector<Prediction> predictions = caffe_model.Classify(final_img);
        string temp = predictions[0].first;
        result = result + temp;
    }
    return result;
}

bool comparestringfornum(string str1, string str2){
    int pos1l = str1.find_last_of('/');
    int pos1r = str1.find_last_of('.');
    int pos2l = str2.find_last_of('/');
    int pos2r = str2.find_last_of('.');
    string n1(str1.substr(pos1l + 1, pos1r - pos1l));
    string n2(str2.substr(pos2l + 1, pos2r - pos2l));
    return atoi(n1.c_str()) < atoi(n2.c_str());
}

bool comparestringforupnum(string str1, string str2){
    int pos1 = str1.find_last_of('/');
    int pos2 = str2.find_last_of('/');
    string n1(str1.substr(pos1 + 1, 3));
    string n2(str2.substr(pos2 + 1, 3));
    return atoi(n1.c_str()) < atoi(n2.c_str());
}

void getRightLabel1(QString labelFilePath, QVector<QString> &labels)
{
    QFile file(labelFilePath);
    if(!file.open(QIODevice::ReadOnly | QIODevice::Text)){
        qDebug() << "Open File Failed";
    }

    QTextStream txtInput(&file);
    QString linestr;
    while(!txtInput.atEnd()){
        linestr = txtInput.readLine();
        labels.append(linestr);
    }
    file.close();
}

int main_cfvdgb(){      //FFT去完澡后不黏连切法完整
    QString right_save_folder = "/home/ubuntu/下载/500_check_data/smallNum/right";
    QString wrong_save_folder = "/home/ubuntu/下载/500_check_data/smallNum/wrong";

    QVector<QString> labels;
    QString labels_path = "/home/ubuntu/下载/500_check_data/smallNum/原图框小写金额/label";
    getRightLabel1(labels_path,labels);

    int right = 0;
    int wrong = 0;
    int index = 0;

    string dir_path;
    dir_path += "/home/ubuntu/桌面/图片/right";
    //dir_path = "/home/ubuntu/下载/500_check_data/smallNum/single";
    vector<string> dirNames;

    ofstream outf;
    outf.open("/home/ubuntu/下载/500_check_data/smallNum/result.txt");
    //outf.open("/home/ubuntu/下载/500_check_data/smallNum/r.txt");
    getAllFiles(dir_path, dirNames);
    sort(dirNames.begin(), dirNames.end(), comparestringfornum);
    Classifier caffe_model = load_caffe_classifier(string("/home/ubuntu/下载/小写数字_model/deploy.prototxt"),
                                                   string("/home/ubuntu/桌面/CNN_Net_iter_3000.caffemodel"),
                                                   string("/home/ubuntu/下载/小写数字_model/label.txt"),
                                                   string("/home/ubuntu/下载/小写数字_model/0-mean.binaryproto"));
    for (int ii = 0; ii < dirNames.size(); ii++){
        Mat img = imread(dirNames[ii]);
        int pos = dirNames[ii].find_last_of('/');
        string name(dirNames[ii].substr(pos + 1));
        Mat fft_thres = get_FFT_threshold(img, false, name, 0.8, 0.7, 2.2);
        Mat i1 = CutNoiseforCompanyName(fft_thres);
        RemoveSmallRegion(i1, i1, 10, 1, 1);
        bitwise_not(i1, i1);
        Mat i2 = i1.clone();
        imshow("df", i1);
       // waitKey();
        vector<Mat> imgs = cutHWLowerNumByH1(i1, "fv", false);
        string text = getNum11(imgs, caffe_model);
        std::cout << dirNames[ii] << " " << text << " " <<labels[index].toStdString() << endl;

        string label_with_point = labels[index].remove("\n").remove("\r").toStdString();
        string label_without_point = label_with_point.substr(0, label_with_point.length() - 3)
                + label_with_point.substr(label_with_point.length() - 2, 2);
        if(text.compare(label_without_point) == 0){
            right++;
            Mat img_ori = imread(dirNames[ii]);
            string path = right_save_folder.toStdString() + "/" + name;
            imwrite(path,img_ori);
        }else{
                    wrong++;
                    Mat img_ori = imread(dirNames[ii]);
                    string path = wrong_save_folder.toStdString() + "/" + name;
                    imwrite(path,img_ori);
                }



        index++;
        outf<<name + "  " + text<<endl;
    }

    cout << right << " " << wrong << endl;
    outf.close();
    return 0;
}



int main_vfbd(){         //tessact方法框完图的识别步骤


    QVector<QString> labels;
    QString labels_path = "/home/ubuntu/桌面/图片/right_label";
    getRightLabel1(labels_path,labels);

    int right = 0;
    int wrong = 0;
    int index = 0;

    string dir_path;
    dir_path += "/home/ubuntu/桌面/图片/right_result";
    vector<string> dirNames;

    ofstream outf;
    outf.open("/home/ubuntu/桌面/图片/result.txt");
    getAllFiles(dir_path, dirNames);
    sort(dirNames.begin(), dirNames.end(), comparestringfornum);
    Classifier caffe_model = load_caffe_classifier(string("/home/ubuntu/下载/小写数字_model/deploy.prototxt"),
                                                   string("/home/ubuntu/桌面/CNN_Net_iter_3000.caffemodel"),
                                                   string("/home/ubuntu/下载/小写数字_model/label.txt"),
                                                   string("/home/ubuntu/下载/小写数字_model/0-mean.binaryproto"));
    for (int ii = 0; ii < dirNames.size(); ii++){
        int pos = dirNames[ii].find_last_of('/');
        string name(dirNames[ii].substr(pos + 1));
        vector<string> filenames;
        getAllFiles(dirNames[ii], filenames);
        sort(filenames.begin(), filenames.end(), comparestringfornum);
        string text = "";
        vector<Mat> imgs;
        for (int j = 0; j < filenames.size(); j++){
            Mat img = imread(filenames[j], 0);
            threshold(img, img, 0, 255, THRESH_OTSU);
            imgs.push_back(img);
        }
        text = text + getNum11(imgs, caffe_model);
        if (text[text.length() - 2] == '0'){        //小写金额数字特征：倒数第二位为0,最后一位为0
            text = text.substr(0, text.length() - 1) + "0";
        }
        std::cout << dirNames[ii] << " " << text << " " <<labels[index].toStdString() << endl;

        string label_with_point = labels[index].remove("\n").remove("\r").toStdString();
        string label_without_point = label_with_point.substr(0, label_with_point.length() - 3)
                + label_with_point.substr(label_with_point.length() - 2, 2);
        if(text.compare(label_without_point) == 0){
            right++;
            outf<<name + "  " + text<<endl;
        }else{
                    wrong++;
                    outf<<name + "  " + text + "     wrong"<<endl;/*
                    Mat wrong_img = imread("/home/ubuntu/桌面/图片/wrong/" + name+".png");
                    string t = "/home/ubuntu/桌面/图片/82/"+name + ".png";
                    imwrite(t, wrong_img);*/
                }



        index++;
    }

    cout << right << " " << wrong << endl;
    outf.close();
}


int main_vfdbd(){             //两种方法合并
    QString right_save_folder = "/home/ubuntu/桌面/图片/all_test/right";
    QString wrong_save_folder = "/home/ubuntu/桌面/图片/all_test/wrong";

    QVector<QString> labels;
    QString labels_path = "/home/ubuntu/桌面/图片/l/label";
    getRightLabel1(labels_path,labels);

    int right = 0;
    int wrong = 0;
    int count1 = 0;
    int count2 = 0;

    string dir_path;
    dir_path += "/home/ubuntu/桌面/图片/l/img";
    //dir_path = "/home/ubuntu/桌面/图片/single";
    vector<string> dirNames;

    ofstream outf;
    outf.open("/home/ubuntu/桌面/图片/all_test/result.txt");
    //outf.open("/home/ubuntu/下载/500_check_data/smallNum/r.txt");
    getAllFiles(dir_path, dirNames);
    sort(dirNames.begin(), dirNames.end(), comparestringfornum);
    Classifier caffe_model = load_caffe_classifier(string("/home/ubuntu/下载/小写数字_model/deploy.prototxt"),
                                                   string("/home/ubuntu/桌面/CNN_Net_iter_3000.caffemodel"),
                                                   string("/home/ubuntu/下载/小写数字_model/label.txt"),
                                                   string("/home/ubuntu/下载/小写数字_model/0-mean.binaryproto"));

    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    api->Init(NULL, "chi_sim");
    for (int ii = 0; ii < dirNames.size(); ii++){
        Mat img = imread(dirNames[ii]);
        Mat img_for_tessact = img.clone();
        int pos = dirNames[ii].find_last_of('/');
        string name(dirNames[ii].substr(pos + 1));
        Mat fft_thres = get_FFT_threshold(img, false, name, 0.855, 0.735, 5.0);
        Mat i1 = CutNoiseforCompanyName(fft_thres);
        RemoveSmallRegion(i1, i1, 10, 1, 1);
        bitwise_not(i1, i1);
        //imshow("df", i1);
       // waitKey();
        vector<Mat> imgs = cutHWLowerNumByH1(i1, "fv", false);
        string text = getNum11(imgs, caffe_model);
        std::cout << dirNames[ii] << " " << text << " " <<labels[ii].toStdString() << endl;

        string label_with_point = labels[ii].remove("\n").remove("\r").toStdString();
        string label_without_point = label_with_point.substr(0, label_with_point.length() - 3)
                + label_with_point.substr(label_with_point.length() - 2, 2);
        if(text.compare(label_without_point) == 0){
            right++;
            Mat img_ori = imread(dirNames[ii]);
            string path = right_save_folder.toStdString() + "/" + name;
            imwrite(path,img_ori);
            outf<<name + "  " + text<<endl;
            count1++;
        }else{
            vector<Mat> imgs_tessact;
            Mat src = img_for_tessact;
            //Mat src = imread("C:\\Users\\Administrator\\Desktop\\wrong_src\\289.png");
            Mat gray = get_FFT_only(src, 0.8, 0.7, 2.2);
            //Mat gray = imread("C:\\Users\\Administrator\\Desktop\\wrong_gray\\289.png", 0);
//            if (ii == 198){
//                cout<<"c"<<endl;
//            }
            vector<int*> coord = getCoord_by_tessact(gray, api);
            for (int i = 0; i < coord.size(); i++) {
                Mat temp = src(Range(coord[i][2], coord[i][3]), Range(coord[i][0], coord[i][1]));
                if (temp.rows > 2 && temp.cols > 2) {
                    vector<Mat> split = get_splitMat(temp);
                    int j = 0;
                    if (i == 0) {
                        j = 1;
                    }
                    for (j; j < split.size(); j++) {
                        split[j] = CutNoise_for_split_downnum(split[j]);
                        RemoveSmallRegion(split[j], split[j], 5, 1, 1);
                        split[j] = cut_margin_for_pic(split[j]);
                        bitwise_not(split[j], split[j]);
                        if (split[j].rows > 2 || split[j].cols > 2) {
                            imgs_tessact.push_back(split[j]);
                        }
                    }
                }
            }
            string text_tessact = getNum11(imgs_tessact, caffe_model);
            if (text_tessact[text_tessact.length() - 2] == '0'){        //小写金额数字特征：倒数第二位为0,最后一位为0
                text_tessact = text_tessact.substr(0, text_tessact.length() - 1) + "0";
            }
            if(text_tessact.compare(label_without_point) == 0){
                right++;
                Mat img_ori = imread(dirNames[ii]);
                string path = right_save_folder.toStdString() + "/" + name;
                imwrite(path,img_ori);
                outf<<name + "  " + text_tessact<<endl;
                count2++;
            }
            else{
                wrong++;
                Mat img_ori = imread(dirNames[ii]);
                string path = wrong_save_folder.toStdString() + "/" + name;
                imwrite(path,img_ori);
                outf<<name + "  " + text + "     wrong"<<endl;
            }
        }
        cout << right << " " << wrong << endl;
    }
    outf.close();
    cout<<endl;
    cout<<count1<<"     "<<count2<<endl;
    return 0;
}

