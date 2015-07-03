// BCDenoising.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "Blockmatching.h"

using namespace cv;
using namespace std;
/**********************************************
* Calculate the PSNR between two images x and y (0-255)
***********************************************/
double PSNR_calculate(Mat x, Mat y)
{
	double psnr = 0.0;
    double sum = 0;
	int sy = x.rows;
	int sx = x.cols;
	for (int i=0; i< sy; i++)
	{
		const uchar* Mx = x.ptr<uchar>(i);
		const double* My = y.ptr<double>(i);
		for (int j=0; j<sx; j++)
			sum += (Mx[j]-My[j])*(Mx[j]-My[j]);
	}
	double div = 255.0*255.0*sx*sy;
	psnr = 10*log10(div/sum);
	return psnr;
}
/**********************************************************
**************** Main function ****************************
Read original image, Gaussian std variance, Generate a noisy image,
Denoise it by BM3D method, output original image, noisy image
and denoised image and display PSNR. 
***********************************************************

Mat img_x : input color original image (Uchar)
Mat gray_x: graysacle original image (Uchar)
Mat img_y : noisy image (double)
Mat out_y : output noisy image (Uchar)
Mat est_y : denoised image (double)
Mat out_est_y : output denoised image (Uchar)

***********************************************************/
int main(int argc, char** argv)
{
	if (argc != 3 )
	{
		cout << "Usage: BlockDenoising.exe img_file sigma" << endl;
		exit(0);
	}

	const char* imagename = argc>1 ? argv[1] : "barbara.png";
	int gau_sigma = argc>2 ? atoi(argv[2]) : 25;

	Ptr<IplImage> iplimg = cvLoadImage(imagename);
	Mat img_x (iplimg);

//	Mat img_x = imread(imagename); 
	Mat gray_x;

	if ( !img_x.data )
		return -1;

	cvtColor(img_x, gray_x, CV_BGR2GRAY);

	Mat noise(gray_x.size(), CV_64FC1);
	randn(noise, Scalar::all(0), Scalar::all(gau_sigma));
	Mat img_y, est_y;
	img_y = Mat::zeros(gray_x.size(), CV_64FC1);
	est_y = Mat::zeros(gray_x.size(), CV_64FC1);
	Mat out_y(img_y.size(), CV_8U);
	Mat out_est_y(img_y.size(), CV_8U);

	for (int i=0; i< img_y.rows; i++)
	{
		const uchar* Mx = gray_x.ptr<uchar>(i);
		const double* Mn = noise.ptr<double>(i);
		double* My = img_y.ptr<double>(i);
		for (int j=0; j<gray_x.cols; j++)
		{
			My[j] = Mn[j]+Mx[j];
		}
	}

	BlockMatching *m_blockmatching;
	m_blockmatching = new BlockMatching(img_y, gau_sigma);
	m_blockmatching->Initial_hardthreshold();
	m_blockmatching->Output_denoised(est_y, out_est_y, out_y);

	double psnr_noisy = PSNR_calculate(gray_x, img_y);
	cout << "The PSNR between noisy and original image is " << psnr_noisy << endl;
	double psnr_denoised = PSNR_calculate(gray_x, est_y);
	cout << "The PSNR between denoised and original image (1st step) is:" << psnr_denoised << endl;
	
	est_y.zeros(img_y.size(), CV_64FC1);
	out_y.zeros(img_y.size(), CV_8U);
	out_est_y.zeros(img_y.size(), CV_8U);

	m_blockmatching->Second_Wiener();
	m_blockmatching->Output_denoised_final(est_y, out_est_y, out_y);

	psnr_denoised = PSNR_calculate(gray_x, est_y);
	cout << "The PSNR between denoised and original image (2nd step) is:" << psnr_denoised << endl;

	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	imshow("Original Image", gray_x);
	namedWindow("Noisy Image", CV_WINDOW_AUTOSIZE);
	imshow("Noisy Image", out_y);
	namedWindow("Denoised Image",CV_WINDOW_AUTOSIZE);
	imshow("Denoised Image", out_est_y);
	waitKey();
	
	delete m_blockmatching;
	noise.release();
	img_y.release();
	est_y.release();
	out_y.release();
	out_est_y.release();
	return 0;
}
