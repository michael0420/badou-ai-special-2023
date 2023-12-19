#include "stdafx.h"
#include<iostream>
#include<random>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

typedef unsigned char       BYTE;
int GaussianNoise(BYTE*InputData, int ImgWidth, int ImgHeight, double PixNumScale)
{
	if (InputData == NULL)
	{
		return -1;
	}
	int PixTotal = ImgWidth*ImgHeight*PixNumScale;

	//高斯随机数
	double MeanValue = 1;
	double Stddev = 2;
	default_random_engine generator;
	normal_distribution<double>dist(MeanValue, Stddev);

	//宽高随机数
	int MinVal = 0;
	int MaxVal_0 = ImgWidth-1;
	int MaxVal_1 = ImgHeight-1;
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int>dis0(MinVal, MaxVal_0);
	uniform_int_distribution<int>dis1(MinVal, MaxVal_1);

	//图像增加高斯噪声
	for (int i = 0; i < PixTotal; i++)
	{
		int x = dis0(gen);
		int y = dis1(gen);
		int PixValue = InputData[y*ImgWidth + x] + dist(generator);
		if(PixValue<0)
		{ 
			PixValue = 0;
		}
		if (PixValue > 255)
		{
			PixValue = 255;
		}
		InputData[y*ImgWidth + x] = PixValue;
	}
	return 0;

}
int main()
{
	string ImgPath = "HomeworkImg.jpg";
	Mat SrcImg = imread(ImgPath);
	Mat GrayImg;
	int ImgWidth = SrcImg.cols;
	int ImgHeight = SrcImg.rows;
	cvtColor(SrcImg, GrayImg, CV_BGR2GRAY);
	namedWindow("GrayImg", 2);
	imshow("GrayImg", GrayImg);
	waitKey(0);

	BYTE*GrayData = new BYTE[ImgWidth*ImgHeight];
	for (int i = 0; i < ImgHeight; i++)
	{
		uchar*data = GrayImg.ptr<uchar>(i);
		for (int j = 0; j < ImgWidth;j++)
		{ 
			GrayData[i*ImgWidth + j] = data[j];
		}
	}
	double PixNumScale = 0.6;
	GaussianNoise(GrayData, ImgWidth, ImgHeight, PixNumScale);
	Mat GaussianNoiseImg(ImgHeight, ImgWidth, CV_8UC1, (unsigned char*)GrayData);
	namedWindow("GaussianNoiseImg", 2);
	imshow("GaussianNoiseImg", GaussianNoiseImg);
	waitKey(0);
	delete[]GrayData;
	return 0;
}
