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
	Mat DstImg = Mat(DstImgSize, CV_8UC3, Scalar(0, 0, 0));
	int i, j;

	//获取三通道数据
	for (i = 0; i < ImgHeight; i++)
	{
		for (j = 0; j < ImgWidth; j++)
		{
			int Bvalue = SrcImg.at<Vec3b>(i, j)[0];
			int Gvalue = SrcImg.at<Vec3b>(i, j)[1];
			int Rvalue = SrcImg.at<Vec3b>(i, j)[2];
			Bdata[i*ImgWidth + j] = Bvalue;
			Gdata[i*ImgWidth + j] = Gvalue;
			Rdata[i*ImgWidth + j] = Rvalue;
		}
	}

	double Xscale =(double) ImgWidth / DstWidth;
	double Yscale = (double)ImgWidth / DstHeight;
	//双线性插值
	for (i = 0; i < DstHeight; i++)
	{
		for (j = 0; j < DstWidth; j++)
		{
			double Xcenter = (j + 0.5)*Xscale - 0.5;
			double Ycenter = (i + 0.5)*Yscale - 0.5;
			int X1 = Xcenter;
			int Y1 = Ycenter;
			int X2 = min(X1 + 1, ImgWidth - 1);
			int Y2 = min(Y1 + 1, ImgHeight - 1);
			double A1 = X2 - Xcenter;
			double A2 = Xcenter - X1;
			double A3 = Y2 - Ycenter;
			double A4 = Ycenter - Y1;
			int Bvalue=A3* (A1*Bdata[Y1*ImgWidth + X1] + A2*Bdata[Y1*ImgWidth + X2]) + A4*(A1* Bdata[Y2*ImgWidth + X1] + A2* Bdata[Y2*ImgWidth + X2]);
			int Gvalue=A3* (A1*Gdata[Y1*ImgWidth + X1] + A2*Gdata[Y1*ImgWidth + X2]) + A4*(A1* Gdata[Y2*ImgWidth + X1] + A2* Gdata[Y2*ImgWidth + X2]);
			int Rvalue=A3* (A1*Rdata[Y1*ImgWidth + X1] + A2*Rdata[Y1*ImgWidth + X2]) + A4*(A1* Rdata[Y2*ImgWidth + X1] + A2* Rdata[Y2*ImgWidth + X2]);
			if (Bvalue < 0)
			{
				Bvalue = 0;
			}
			if (Bvalue > 255)
			{
				Bvalue = 255;
			}
			if (Gvalue < 0)
			{
				Gvalue = 0;
			}
			if (Gvalue > 255)
			{
				Gvalue = 255;
			}
			if (Rvalue < 0)
			{
				Rvalue = 0;
			}
			if (Rvalue > 255)
			{
				Rvalue = 255;
			}
			
			DstImg.at<Vec3b>(i, j) = Vec3b(Bvalue, Gvalue, Rvalue);
		}
	}

	imwrite("BilinearInterpolation.jpg", DstImg);
	namedWindow("DstImg", 2);
	imshow("DstImg", DstImg);
	waitKey(0);
	delete[]Bdata;
	delete[]Gdata;
	delete[]Rdata;
	return 0;
}
