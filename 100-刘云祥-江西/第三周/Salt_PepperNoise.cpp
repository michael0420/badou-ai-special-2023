#include "stdafx.h"
#include<iostream>
#include<random>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

typedef unsigned char       BYTE;
int SaltPepperNoise(BYTE*InputData, int ImgWidth, int ImgHeight, double PixNumScale)
{
	if (InputData == NULL)
	{
		return -1;
	}
	int PixTotal = ImgWidth*ImgHeight*PixNumScale;

	//宽高随机数
	int MinVal = 0;
	int MaxVal_0 = ImgWidth - 1;
	int MaxVal_1 = ImgHeight - 1;
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int>dis0(MinVal, MaxVal_0);
	uniform_int_distribution<int>dis1(MinVal, MaxVal_1);

	//图像增加椒盐噪声
	for (int i = 0; i < PixTotal; i++)
	{
		int x = dis0(gen);
		int y = dis1(gen);
		
		//判定为0或255的方法可自由选择
		if (x%2==0)  
		{
			InputData[y*ImgWidth+x]=255;
		}
		else
		{
			InputData[y*ImgWidth + x] = 0;
		}

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
		for (int j = 0; j < ImgWidth; j++)
		{
			GrayData[i*ImgWidth + j] = data[j];
		}
	}
	double PixNumScale = 0.1;
	SaltPepperNoise(GrayData, ImgWidth, ImgHeight, PixNumScale);
	Mat GaussianNoiseImg(ImgHeight, ImgWidth, CV_8UC1, (unsigned char*)GrayData);
	namedWindow("GaussianNoiseImg", 2);
	imshow("GaussianNoiseImg", GaussianNoiseImg);
	waitKey(0);
	delete[]GrayData;
	return 0;
}
