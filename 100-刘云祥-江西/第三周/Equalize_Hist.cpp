#include "stdafx.h"
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
typedef unsigned char       BYTE;

int EqualizeHist(BYTE*InputData, BYTE*OutputData, int ImgWidth, int ImgHeight)
{
	int GrayValue[256] = { 0 };           //记录不同灰度级下的像素个数
	double GrayProd[256] = { 0 };         //记录灰度密度
	double GrayDistribution[256] = { 0 }; //记录累计灰度密度
	int GrayEqualize[256] = { 0 };        //记录均衡化灰度值

	for (int i = 0; i < ImgHeight; i++)
	{
		for (int j = 0; j < ImgWidth; j++)
		{
			int Value = InputData[i*ImgWidth + j];
			GrayValue[Value]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		GrayProd[i] = (double)GrayValue[i] / (ImgWidth*ImgHeight);
		if (i == 0)
		{
			GrayDistribution[0] = GrayProd[0];
		}
		else
		{
			GrayDistribution[i] = GrayDistribution[i - 1] + GrayProd[i];
		}
	}


	for (int i = 0; i < 256; i++)
	{
		GrayEqualize[i] = (GrayDistribution[i] * 256 - 1) + 0.5;
	}

	for (int i =0 ; i < ImgHeight; i++)
	{
		for (int j = 0; j < ImgWidth; j++)
		{
			int Value = InputData[i*ImgWidth + j];
			OutputData[i*ImgWidth + j] = GrayEqualize[Value];
		}
	}

	return 0;
}


int main()
{
	Mat GrayImg;
	Mat SrcImg = imread("HomeworkImg.jpg");
	if (SrcImg.data == NULL)
	{
		return -1;
	}
	cvtColor(SrcImg, GrayImg, CV_BGR2GRAY);
	namedWindow("GrayImg", 2);
	imshow("GrayImg", GrayImg);
	waitKey(0);

	int ImgWidth = GrayImg.cols;
	int ImgHeight = GrayImg.rows;
	BYTE *GrayImgData = new BYTE[ImgWidth*ImgHeight];
	BYTE *EqualizeHistImgData = new BYTE[ImgWidth*ImgHeight];
	int k = 0;
	for (int i = 0; i < ImgHeight; i++)
	{
		uchar*data = GrayImg.ptr<uchar>(i);
		for (int j = 0; j < ImgWidth; j++)
		{
			GrayImgData[k++] = data[j];
		}
	}
	EqualizeHist(GrayImgData, EqualizeHistImgData, ImgWidth, ImgHeight);
	Mat EqualizeHistImg(ImgHeight, ImgWidth, CV_8UC1, (unsigned char*)EqualizeHistImgData);
	namedWindow("EqualizeHistImg", 2);
	imshow("EqualizeHistImg", EqualizeHistImg);
	waitKey(0);
  delete[]GrayImgData;
  delete[]EqualizeHistImgData;
  return 0;
}
