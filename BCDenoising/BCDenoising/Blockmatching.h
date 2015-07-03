#ifndef _BLOCKMATCHING_H

#define _BLOCKMATCHING_H

#include <iostream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"

using namespace cv;
using namespace std;
class BlockMatching
{
private:
	int width;
	int height;
	double gau_sigma;
	Mat noisy_y;
	Mat est_y;
	Mat est_final;

	int Hd_Nstep;       // sliding step to process every next refernece block
	int Hd_N2;           // maximum number of similar blocks (maximum 3rd dimension size of a 3D block)
	int Hd_Ns;           // search window's length of the side
	int Hd_beta;         // parameter of the 2D Kaiser window
	double tau_match;    // threshold for the block distance (d-distance)
	double lambda_thr2D; // threshold parameter for the coarse initial denoising used in the d-distance measure
	double lambda_thr3D; // threshold parameter for the hard-thresholding in 3D DFT domain

	int Wiener_Nstep; 
	int Wiener_N2;
	int Wiener_Ns;
	int Wiener_beta; 

	int Hd_N1; 
	int Wiener_N1;
	double tau_match_wiener;
	int Hd_Ns_half;
	int Wiener_Ns_half;

	Mat Hd_Wwin2D;
	Mat Wiener_Wwin2D;

public:
	BlockMatching(Mat in_noisy_y, int in_sigma);
	virtual ~BlockMatching();
	void Output_denoised(Mat &dout_est_y, Mat &out_est_y, Mat &out_noisy_y);
	void Output_denoised_final(Mat &dout_est_y, Mat &out_est_y, Mat &out_noisy_y);
	void Initial_hardthreshold();
	inline uchar PixelRange(double a0, int m0, int m1);
	inline int Calculate_minmax(int a0, int m0, int m1);
	int SimilarPatch_Ordering(Mat* tau_fft, Point index_cur, Point index_min,  Point index_search, double lambda, int *&SR_order, Mat *&Group_ID);
	void Hardthreshold( Mat& in_matrix, double lambda);
	double Distance_vector( Mat a1, Mat a2);
	double Distance_wiener ( Mat a1, Mat a2);
	double Denoising_hardthreshold( Mat* group_trans, int totalnum, double lambda, Mat *&group);
	void Aggregation_hardthreshold ( Mat& ebuff, Mat& wbuff, int totalnum, double weight_xR, Mat *group, int *SR_order);
	void Second_Wiener();
	int Wiener_SimilarPatch_Ordering(Point index_cur, Point index_min,  Point index_search,  int *&SR_order, Mat *&Group_est, Mat *&Group_origin ); 
    double Denoising_wiener( Mat* group_est, Mat* group_origin, int totalnum, Mat *&group);
	void Aggregation_wiener ( Mat& ebuff, Mat& wbuff, int totalnum, double weight_xR, Mat *group, int *SR_order);
	void TransformPadding ( Mat& tBlock, int Size );
};
#endif