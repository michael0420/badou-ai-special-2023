int main()
{
	string ImgPath = "HomeworkImg.jpg";
	Mat SrcImg = imread(ImgPath);
	if (SrcImg.data == NULL)
	{
		return -1;
	}
	int ImgWidth = SrcImg.cols;
	int ImgHeight = SrcImg.rows;
	char *GrayImgData = new char[ImgWidth*ImgHeight];
	char *BinaryImgData = new char[ImgWidth*ImgHeight];
	int i, j;
	//灰度图像
	for (i = 0; i < ImgHeight; i++)
	{
		for (j = 0; j < ImgWidth; j++)
		{
			int Bvalue = SrcImg.at<Vec3b>(i, j)[0];
			int Gvalue = SrcImg.at<Vec3b>(i, j)[1];
			int Rvalue = SrcImg.at<Vec3b>(i, j)[2];
			GrayImgData[i*ImgWidth + j] = (Bvalue*0.114 + Gvalue*0.587 + Rvalue*0.299)+0.5;
		}
	}
	Mat GrayImg(ImgHeight, ImgWidth, CV_8UC1, (unsigned char*)GrayImgData);
	namedWindow("GrayImg", 2);
	imshow("GrayImg", GrayImg);
	waitKey(0);

	//二值化图像
	for (i = 0; i < ImgHeight; i++)
	{
		for (int j = 0; j < ImgWidth; j++)
		{
			if (GrayImgData[i*ImgWidth + j]>100)
			{
				BinaryImgData[i*ImgWidth + j] = 255;
			}
			else
			{
				BinaryImgData[i*ImgWidth + j] = 0;
			}
		}
	}
	Mat BinaryImg(ImgHeight, ImgWidth, CV_8UC1, (unsigned char*)BinaryImgData);
	namedWindow("BinaryImg", 2);
	imshow("BinaryImg", BinaryImg);
	waitKey(0);
	delete[]GrayImgData;
	delete[]BinaryImgData;
    return 0;
}
