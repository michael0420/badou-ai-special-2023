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
	BYTE *Bdata = new BYTE[ImgWidth*ImgHeight];
	BYTE *Gdata = new BYTE[ImgWidth*ImgHeight];
	BYTE *Rdata = new BYTE[ImgWidth*ImgHeight];
	int DstWidth = 800;
	int DstHeight = 800;
	Size DstImgSize;
	DstImgSize.width = DstWidth;
	DstImgSize.height = DstHeight;
	Mat DstImg=Mat(DstImgSize, CV_8UC3, Scalar(0, 0, 0));
	double Xratio =(double) ImgWidth / DstWidth;
	double Yratio =(double) ImgHeight / DstHeight;
	int i, j;
	//获取三通道数据
	for (i = 0; i < ImgHeight; i++)
	{
		for (j = 0; j < ImgWidth; j++)
		{
			int Bvalue = SrcImg.at<Vec3b>(i, j)[0];
			int Gvalue = SrcImg.at<Vec3b>(i, j)[1];
			int Rvalue = SrcImg.at<Vec3b>(i, j)[2];
			Bdata[i*ImgWidth+j] = Bvalue;
			Gdata[i*ImgWidth + j] = Gvalue;
			Rdata[i*ImgWidth + j] = Rvalue;
		}
	}
	//邻近插值
	for (i = 0; i < DstHeight; i++)
	{
		for (j = 0; j < DstWidth; j++)
		{
			int X = (i *Yratio + 0.5);
			int Y = (j* Xratio + 0.5);
			DstImg.at<Vec3b>(i, j) = Vec3b(Bdata[X*ImgWidth + Y], Gdata[X*ImgWidth + Y], Rdata[X*ImgWidth + Y]);
		}
	}

	imwrite("NearestInterpolationImg.jpg", DstImg);
	namedWindow("DstImg", 2);
	imshow("DstImg", DstImg);
	waitKey(0);
	delete[]Bdata;
	delete[]Gdata;
	delete[]Rdata;
	return 0;
}
