#include "stdafx.h"
#include <vector>
#include <algorithm>
#include "Blockmatching.h"
#include "RelatedOperations.h"

struct myclass 
{
	bool operator() ( Vec2d i, Vec2d j) { return ( i[0]<j[0] );}
}myobject;

BlockMatching::BlockMatching(Mat in_noisy_y, int in_sigma)
{
	width = in_noisy_y.cols;
	height = in_noisy_y.rows;
	double scale = 1./255;
	gau_sigma = (double)in_sigma*scale;
	noisy_y = in_noisy_y * scale;
	est_y = Mat::zeros(noisy_y.size(), CV_64FC1);
	est_final = Mat::zeros(noisy_y.size(), CV_64FC1);

	Hd_Nstep = 4;     
	Hd_N2 = 28;           
	Hd_Ns = 73;           
	Hd_beta = 4;         
	tau_match = 0.233;    
	lambda_thr2D = 0.82; 
	lambda_thr3D = 0.75; 

	Wiener_Nstep = 3;
	Wiener_N2 = 72;
	Wiener_Ns = 35;
	Wiener_beta = 3;

	Hd_N1 = (int)PixelRange( in_sigma*0.08+9.4, 8, 13);
	Hd_N1 = Hd_N1 - 1 + Hd_N1%2;
	Wiener_N1 = (int)PixelRange( in_sigma*0.15+5.625, 7, 11);
	Wiener_N1 = Wiener_N1 - 1 + Wiener_N1%2;
	tau_match_wiener = (double)in_sigma/4000 + 0.0105;
	
	Hd_Ns_half = (Hd_Ns-1)/2;
	Wiener_Ns_half = (Wiener_Ns-1)/2;
	tau_match = tau_match * Hd_N1;
	tau_match_wiener = tau_match_wiener*Wiener_N1;

	Hd_Wwin2D = Mat::zeros(Size(Hd_N1,Hd_N1),CV_64FC1);
	Wiener_Wwin2D = Mat::zeros(Size(Wiener_N1, Wiener_N1), CV_64FC1);
	RelatedOperations m_operation;
	m_operation.Kaiser_Window(Hd_Wwin2D, Hd_N1, (double)Hd_beta);
	m_operation.Kaiser_Window(Wiener_Wwin2D, Wiener_N1, (double)Wiener_beta);
}

BlockMatching::~BlockMatching()
{
	noisy_y.release();
	est_y.release();
	est_final.release();
	Hd_Wwin2D.release();
	Wiener_Wwin2D.release();
}
/**********************************************
* Output the denoised image from 1st step (double,
  uchar) and the noisy image (uchar) for display 
***********************************************/
void BlockMatching::Output_denoised(Mat &dout_est_y, Mat &out_est_y, Mat &out_noisy_y)
{
	for (int i=0; i<height; i++)
	{
		double* My = est_y.ptr<double>(i);
		double* DMy = dout_est_y.ptr<double>(i);
		uchar* Oy = out_est_y.ptr<uchar>(i);
		double* Ny = noisy_y.ptr<double>(i);
		uchar* ONy = out_noisy_y.ptr<uchar>(i);
		for (int j=0; j<width; j++)
		{
			DMy[j] = My[j]*255;
			Oy[j] = PixelRange(My[j]*255,0,255);
			ONy[j] = PixelRange(Ny[j]*255, 0, 255);
		}
	}
}
/**********************************************
* Output the denoised image from 2nd step (double,
  uchar) and the noisy image (uchar) for display 
***********************************************/
void BlockMatching::Output_denoised_final(Mat &dout_est_y, Mat &out_est_y, Mat &out_noisy_y)
{
	for (int i=0; i<height; i++)
	{
		double* My = est_final.ptr<double>(i);
		double* DMy = dout_est_y.ptr<double>(i);
		uchar* Oy = out_est_y.ptr<uchar>(i);
		double* Ny = noisy_y.ptr<double>(i);
		uchar* ONy = out_noisy_y.ptr<uchar>(i);
		for (int j=0; j<width; j++)
		{
			DMy[j] = My[j]*255;
			Oy[j] = PixelRange(My[j]*255,0,255);
			ONy[j] = PixelRange(Ny[j]*255, 0, 255);
		}
	}
}
/**********************************************
* output the Pixel value a0 within range (m0, m1)
***********************************************/
inline uchar BlockMatching::PixelRange(double a0, int m0, int m1)
{
	uchar a = (uchar)ceil(a0+0.5);
	uchar b = ( a < m0 ? m0:a)>m1 ? m1 : ( a < m0 ? m0:a);
	return b;
}
/**********************************************
* output the int value a0 within range (m0, m1)
***********************************************/
inline int BlockMatching::Calculate_minmax(int a0, int m0, int m1)
{
	int b = ( a0 < m0 ? m0:a0)>m1 ? m1 : ( a0 < m0 ? m0:a0);
	return b;
}
/**********************************************
* Pad OpenCV DFT result 
***********************************************/
void BlockMatching::TransformPadding( Mat& tBlock, int Size )
{
	int Half = (Size-1)/2;
	for ( int i=0; i<Size; i++)
	{
		Vec2d* cur = tBlock.ptr<Vec2d>(i);
		if (i==0)
		{
			for ( int j=Half+1; j<Size; j++)
			{
				cur[j].val[0] = cur[Size-j].val[0];
				cur[j].val[1] = -cur[Size-j].val[1];
			}
		}
		else
		{
			Vec2d* cur1 = tBlock.ptr<Vec2d>(Size-i);
			for ( int j=Half+1; j<Size; j++)
			{
				cur[j].val[0] = cur1[Size-j].val[0];
				cur[j].val[1] = -cur1[Size-j].val[1];
			}
		}
	}
}
/**********************************************
* BM3D first step: group similar blocks and use a 
collaborative 3D filter and hard thresholding the 
coefficients to obtain an initial denoised image
***********************************************/
void BlockMatching::Initial_hardthreshold()
{
	int N1_half = (Hd_N1 - 1)/2;
	double lambda_thr2D_total = lambda_thr2D*gau_sigma*sqrt(2*log((double)(Hd_N1*Hd_N1)));
	double lambda_thr3D_total = lambda_thr3D*gau_sigma*sqrt(2*log((double)(Hd_N1*Hd_N1)));
	int col_xR = width - (2*N1_half);
	int row_xR = height - (2*N1_half);
	double scale = 1./Hd_N1;

	Mat ebuff, wbuff, cBlock, tBlock;
	ebuff = Mat::zeros(Size(width, height), CV_64FC1);
	wbuff = Mat::zeros(Size(width, height), CV_64FC1);
	cBlock = Mat::zeros(Size(Hd_N1, Hd_N1), CV_64FC1);
	tBlock = Mat::zeros(Size(Hd_N1, Hd_N1), CV_64FC2);
	Mat* tau_Zx4D = new Mat[row_xR*col_xR];

	for ( int i=0; i<row_xR; i++)
		for (int j=0; j<col_xR; j++)
		{
			noisy_y( Range(i, i+(2*N1_half)+1), Range(j, j+(2*N1_half)+1)).copyTo(cBlock);
		    dft(cBlock, tBlock, DFT_COMPLEX_OUTPUT, 0);
			TransformPadding ( tBlock, Hd_N1 );
			tBlock = tBlock * scale;
			tBlock.copyTo( tau_Zx4D[i*col_xR+j] );
		}

	if( (row_xR-1)%Hd_Nstep !=0 )
		row_xR = (int)(( floor((double)(row_xR-1)/Hd_Nstep)+1)*Hd_Nstep + 1);
	if ( (col_xR-1)%Hd_Nstep !=0 )
		col_xR = (int)(( floor((double)(col_xR-1)/Hd_Nstep)+1)*Hd_Nstep + 1);

	int i_center, j_center, i_min, i_max, j_min, j_max;
	int *SR_order = new int [Hd_N2]; 

	Mat* group = new Mat[Hd_N2];
	Mat* group_trans = new Mat [Hd_N2];
	for ( int k=0; k<Hd_N2; k++)
	{
		group[k].create(Hd_N1, Hd_N1, CV_64FC1);
		group_trans[k].create(Hd_N1, Hd_N1, CV_64FC2);
	}
	int foundnum = 0;
	double weight_xR = 1.0;
	for ( int i=0; i<row_xR; i=i+Hd_Nstep)
	{
		if ( i> height-(2*N1_half)-1 )
			i = height-(2*N1_half)-1;
		for ( int j=0; j<col_xR; j=j+Hd_Nstep)
		{
			if ( j>width-(2*N1_half)-1 )
				j = width-(2*N1_half)-1;
			
			i_center = i+N1_half;
			j_center = j+N1_half;
			i_min = Calculate_minmax( i_center-Hd_Ns_half, 0, height-1);
			i_max = Calculate_minmax( i_center+Hd_Ns_half, 0, height-1);
			j_min = Calculate_minmax( j_center-Hd_Ns_half, 0, width-1);
			j_max = Calculate_minmax( j_center+Hd_Ns_half, 0, width-1);

			Point index_cur( j, i );
			Point index_Nsmin( j_min, i_min);
			Point index_search( j_max-j_min+1, i_max-i_min+1 );
			
			foundnum = SimilarPatch_Ordering(tau_Zx4D, index_cur, index_Nsmin, index_search, lambda_thr2D_total, SR_order, group_trans);
			weight_xR = Denoising_hardthreshold(group_trans, foundnum, lambda_thr3D_total, group);
			Aggregation_hardthreshold( ebuff, wbuff, foundnum, weight_xR, group, SR_order);
		}
	}

	for ( int i=0; i<height; i++)
		for ( int j=0; j<width; j++)
		{
			if ( wbuff.at<double>(i,j)==0 )
				wbuff.at<double>(i,j) = 1;
		}
    est_y = ebuff/wbuff;

	ebuff.release();
	wbuff.release();
	cBlock.release();
	tBlock.release();
	delete [] SR_order;
	for ( int i=0; i< height - (2*N1_half); i++)
		for (int j=0; j< width - (2*N1_half); j++)
			tau_Zx4D[i*(width-2*N1_half)+j].release();
	delete [] tau_Zx4D;
	for (int k=0; k<Hd_N2; k++)
	{
		group[k].release();
		group_trans[k].release();
	}
	delete [] group;
	delete [] group_trans;
}
/**********************************************
* Find the similar blocks of the current prossed block
* tau_fft: FFT of each N1xN1 block in the whole image
* index_cur: left corner of current processed block 
* index_min: left corner of searching window
* index_search: size of searching window
* lambda: input lambda_thr2D_total
* SR_order: output 1D MxN array with order index
* Group_ID: similar blocks 2D transform version
* return number of similar blocks
***********************************************/
int BlockMatching::SimilarPatch_Ordering( Mat* tau_fft, Point index_cur, Point index_min, Point index_search, double lambda, int *&SR_order, Mat *&Group_ID)
{
	int i_min = index_min.y;
	int j_min = index_min.x;
	int search_y = index_search.y;
	int search_x = index_search.x;
	int N1_half = (Hd_N1 - 1)/2;
	int col_xR = width - (2*N1_half);

	for (int k=0; k<Hd_N2; k++)
		Group_ID[k].zeros(Hd_N1, Hd_N1, CV_64FC2);

	Mat SxR = Mat::ones( height, width, CV_64FC1)*(-1.0);
	memset( SR_order, -1, sizeof(int)*Hd_N2 );
	Mat T_cur = Mat::ones(Hd_N1, Hd_N1, CV_64FC1);
	tau_fft[index_cur.y*col_xR+index_cur.x].copyTo(T_cur);	
	Hardthreshold(T_cur, lambda);

	int index_y, index_x;
	double distance = 0.0;
	Mat T_ser = Mat::ones(Hd_N1, Hd_N1, CV_64FC1);
	for (int i=0; i<search_y-2*N1_half; i++)
		for (int j=0; j<search_x-2*N1_half; j++)
		{
			index_y = i+i_min;
			index_x = j+j_min;
			tau_fft[index_y*col_xR+index_x].copyTo(T_ser);
			Hardthreshold(T_ser, lambda);

			distance = Distance_vector(T_cur, T_ser);
			if ( distance < tau_match )
				SxR.at<double>(index_y, index_x) = distance;
		}

    vector<Vec2d> myvector;
	double coeff=0.0, pos=0.0;
	for (int i=0; i<height; i++)
		for (int j=0; j<width; j++)
		{
			coeff = SxR.at<double>(i,j);
			pos = (double)(i*width + j);
			if ( coeff >= 0)
				myvector.push_back( Vec2d(coeff, pos) );
		}
	sort ( myvector.begin(), myvector.end(), myobject );
	
	int foundnum = myvector.size();
	foundnum = (foundnum<Hd_N2) ? foundnum : Hd_N2;
	Mat cBlock = Mat::zeros(Size(Hd_N1, Hd_N1), CV_64FC2);
	int found_pos=0, pos_x=0, pos_y=0;
	for (int k=0; k<foundnum; k++)
	{
		Vec2d it = myvector.at(k);
		found_pos = (int)( it[1] );
		pos_y = found_pos/width;
		pos_x = found_pos%width;
		cBlock = tau_fft[pos_y*col_xR+pos_x]*((double)Hd_N1);
		cBlock.copyTo(Group_ID[k]);
		SR_order[k] = found_pos;
	}
		
	SxR.release();
	myvector.clear();
	cBlock.release();
	T_cur.release();
	T_ser.release();
	return foundnum;
}
/**********************************************
* Hard-thresholding the 2D block with input lambda
***********************************************/
void BlockMatching::Hardthreshold(Mat& in_matrix, double lambda)
{
	double mag=0.0;
	for (int i=0; i<in_matrix.rows; i++)
		for (int j=0; j<in_matrix.cols; j++)
		{
			Vec2d Pix = in_matrix.at<Vec2d>(i,j);
			mag = sqrt(Pix[0]*Pix[0]+Pix[1]*Pix[1]);
			if (mag <= lambda )
				in_matrix.at<Vec2d>(i,j) = Vec2d(0,0);
		}
}
/**********************************************
* The mean squre error between two real matrix a1, a2
***********************************************/
double BlockMatching::Distance_vector(Mat a1, Mat a2)
{
	double disum = 0.0;
	int in_row = a1.rows;
	int in_col = a1.cols;
	Vec2d v_a1, v_a2;
	for (int i=0; i<in_row; i++)
		for (int j=0; j<in_col; j++)
		{
			v_a1 = a1.at<Vec2d>(i,j);
			v_a2 = a2.at<Vec2d>(i,j);
			disum += pow( (v_a1[0]-v_a2[0]), 2) + pow( (v_a1[1]-v_a2[1]), 2);
		}
	disum = sqrt(disum);
	return disum;
}
/**********************************************
* The distance calculation in wiener filtering
***********************************************/
double BlockMatching::Distance_wiener(Mat a1, Mat a2 )
{
	Mat r1 = Mat::zeros(Wiener_N1, Wiener_N1, CV_64FC1);
	Mat r2 = Mat::zeros(Wiener_N1, Wiener_N1, CV_64FC1);
	a1.copyTo(r1);
	a2.copyTo(r2);

	double distance = 0.0;
	double sum1 = 0.0, sum2 = 0.0;
	for ( int i=0; i<Wiener_N1; i++)
		for ( int j=0; j<Wiener_N1; j++)
		{
			sum1 += r1.at<double>(i,j);
			sum2 += r2.at<double>(i,j);
		}
	sum1 = sum1/(Wiener_N1*Wiener_N1);
	sum2 = sum2/(Wiener_N1*Wiener_N1);

	double diff;
	for ( int i=0; i<Wiener_N1; i++)
		for ( int j=0; j<Wiener_N1; j++)
		{
			diff = r1.at<double>(i,j) - sum1 - r2.at<double>(i,j) + sum2;
			distance += diff*diff;
		}

	distance = sqrt(distance);
	r1.release();
	r2.release();
	return distance;
}
/**********************************************
* Hard_thresholding denoising within a 3D group
* group_trans: 2D transformed version of similar blocks
* totalnum: number of similar blocks
* lambda: input 3D hard threshold
* group: output inversed blocks of the group
* return the weight for next aggregation
***********************************************/
double BlockMatching::Denoising_hardthreshold(Mat* group_trans, int totalnum, double lambda, Mat *&group)
{
	double scale = 1./sqrt( (double)Hd_N1*Hd_N1*totalnum );
	double weight_xR = 1.0;

	for (int k=0; k<totalnum; k++)
		group[k].zeros(Hd_N1, Hd_N1, CV_64FC1);

	Mat time_in = Mat::zeros(totalnum, 1, CV_64FC2);
	Mat time_trans = Mat::zeros(totalnum, 1, CV_64FC2);
	int statis_nonzero = 0;
	double mag = 0.0;
	for (int i=0; i<Hd_N1; i++)
		for (int j=0; j<Hd_N1; j++)
		{
			for (int k=0; k<totalnum; k++)
			{
				time_in.at<Vec2d>(k,0) = group_trans[k].at<Vec2d>(i,j);
			}

			dft( time_in, time_trans, DFT_COMPLEX_OUTPUT, 0 );
			
			for (int k=0; k<totalnum; k++)
			{
				Vec2d Pix = time_trans.at<Vec2d>(k,0);

				Pix[0] = Pix[0] * scale;
				Pix[1] = Pix[1] * scale;
				mag = sqrt(Pix[0]*Pix[0]+Pix[1]*Pix[1]);
				if (mag <= lambda )
					time_trans.at<Vec2d>(k,0) = Vec2d(0,0);
				else
				{
					time_trans.at<Vec2d>(k,0) = Vec2d( Pix[0], Pix[1] );
					statis_nonzero++;
				}
			}
	
			dft( time_trans, time_in, DFT_INVERSE + DFT_COMPLEX_OUTPUT + DFT_SCALE, 0 );
			for (int k=0; k<totalnum; k++)
			{
				group_trans[k].at<Vec2d>(i,j) = time_in.at<Vec2d>(k,0);
			}
		}

    for ( int k=0; k<totalnum; k++)
	{
		Mat inver_coeff = group_trans[k];
		dft( inver_coeff, group[k], DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE, 0 );
		group[k] = group[k]/scale;
	}

	if (statis_nonzero>=1)
		weight_xR = 1./statis_nonzero;

	time_in.release();
	time_trans.release();
	return weight_xR;
}
/**********************************************
* Aggregation among the inversed denoised block back
  to the original position
* totalnum: number of similar blocks
* weight_xR: weight for next aggregation
* group: input inversed blocks of the group
* SR_order: input 1D MxN array with order index
***********************************************/
void BlockMatching::Aggregation_hardthreshold( Mat& ebuff, Mat& wbuff, int totalnum, double weight_xR, Mat *group, int *SR_order)
{
	Mat W_xR = Mat::zeros( Hd_N1, Hd_N1, CV_64FC1);
	Hd_Wwin2D.copyTo(W_xR);
	W_xR = W_xR * weight_xR;

	Mat temp1 = Mat::zeros( Hd_N1, Hd_N1, CV_64FC1);
	int found_pos = 0, found_i=0, found_j=0;
	for ( int k=0; k<totalnum; k++)
	{
		temp1 = group[k].mul(W_xR);
		found_pos = SR_order[k];
		found_i = found_pos/width;
		found_j = found_pos%width;
		for ( int i=0; i<Hd_N1; i++)
		{
			double* Mi = temp1.ptr<double>(i);
			double* Ni = ebuff.ptr<double>(i+found_i);
			for ( int j=0; j<Hd_N1; j++)
				Ni[j+found_j] += Mi[j];
		}
		for ( int i=0; i<Hd_N1; i++)
		{
			double* Mi = W_xR.ptr<double>(i);
			double* Ni = wbuff.ptr<double>(i+found_i);
			for ( int j=0; j<Hd_N1; j++)
				Ni[j+found_j] += Mi[j];
		}
	}

	W_xR.release();
	temp1.release();
}
/**********************************************
* BM3D second step: group similar blocks and use a 
collaborative 3D filter and Wiener filtering the 
coefficients to obtain a final denoised image based 
on the denoised result from the first step
************************************************/
void BlockMatching::Second_Wiener()
{
	int N1_half = (Wiener_N1 - 1)/2;

	int col_xR = width - (2*N1_half);
	int row_xR = height - (2*N1_half);
	if( (row_xR-1)%Wiener_Nstep !=0 )
		row_xR = (int)(( floor((double)(row_xR-1)/Wiener_Nstep)+1)*Wiener_Nstep + 1);
	if ( (col_xR-1)%Wiener_Nstep !=0 )
		col_xR = (int)(( floor((double)(col_xR-1)/Wiener_Nstep)+1)*Wiener_Nstep + 1);

	Mat ebuff, wbuff;
	ebuff = Mat::zeros(Size(width, height), CV_64FC1);
	wbuff = Mat::zeros(Size(width, height), CV_64FC1);

	int i_center, j_center, i_min, i_max, j_min, j_max;
	int *SR_order = new int [Wiener_N2]; 
	Mat* group_est = new Mat[Wiener_N2];
	Mat* group_origin = new Mat [Wiener_N2];
	Mat* group_denoised = new Mat [Wiener_N2];
	for ( int k=0; k<Wiener_N2; k++)
	{
		group_est[k].create(Wiener_N1, Wiener_N1, CV_64FC1);
		group_origin[k].create(Wiener_N1, Wiener_N1, CV_64FC1);
		group_denoised[k].create(Wiener_N1, Wiener_N1, CV_64FC1);
	}
	int foundnum = 0;
	double weight_xR = 1.0;
	for ( int i=0; i<row_xR; i=i+Wiener_Nstep)
	{
		if ( i> height-(2*N1_half)-1 )
			i = height-(2*N1_half)-1;
		for ( int j=0; j<col_xR; j=j+Wiener_Nstep)
		{
			if ( j>width-(2*N1_half)-1 )
				j = width-(2*N1_half)-1;
			
			i_center = i+N1_half;
			j_center = j+N1_half;
			i_min = Calculate_minmax( i_center-Wiener_Ns_half, 0, height-1);
			i_max = Calculate_minmax( i_center+Wiener_Ns_half, 0, height-1);
			j_min = Calculate_minmax( j_center-Wiener_Ns_half, 0, width-1);
			j_max = Calculate_minmax( j_center+Wiener_Ns_half, 0, width-1);

			Point index_cur( j, i );
			Point index_Nsmin( j_min, i_min);
			Point index_search( j_max-j_min+1, i_max-i_min+1 );
			
			foundnum = Wiener_SimilarPatch_Ordering(index_cur, index_Nsmin, index_search, SR_order, group_est, group_origin);
			weight_xR = Denoising_wiener( group_est, group_origin, foundnum, group_denoised );
			Aggregation_wiener( ebuff, wbuff, foundnum, weight_xR, group_denoised, SR_order);
		}
	}

	for ( int i=0; i<height; i++)
		for ( int j=0; j<width; j++)
		{
			if ( wbuff.at<double>(i,j)==0 )
				wbuff.at<double>(i,j) = 1;
		}
    est_final = ebuff/wbuff;

	ebuff.release();
	wbuff.release();
	delete [] SR_order;
	for (int k=0; k<Wiener_N2; k++)
	{
		group_est[k].release();
		group_origin[k].release();
		group_denoised[k].release();
	}
	delete [] group_est;
	delete [] group_origin;
	delete [] group_denoised;
}
/**********************************************
* Find the similar blocks of the current prossed block 
  in the second step
* index_cur: left corner of current processed block 
* index_min: left corner of searching window
* index_search: size of searching window
* cBlock: the current block on the estiamted image
* SR_order: output 1D array with group order index
* group_est: similar blocks 2D spatial version on the
             denosied image from the first step
* group_est: similar blocks 2D spatial version on the
             noisy image
* return number of similar blocks
***********************************************/
int BlockMatching::Wiener_SimilarPatch_Ordering(Point index_cur, Point index_min,  Point index_search, int *&SR_order, Mat *&Group_est, Mat *&Group_origin )
{
	int i_min = index_min.y;
	int j_min = index_min.x;
	int search_y = index_search.y;
	int search_x = index_search.x;
	int N1_half = (Wiener_N1 - 1)/2;

	for (int k=0; k<Wiener_N2; k++)
	{
		Group_est[k].zeros(Wiener_N1, Wiener_N1, CV_64FC1);
		Group_origin[k].zeros(Wiener_N1, Wiener_N1, CV_64FC1);
	}

	Mat SxR = Mat::ones( height, width, CV_64FC1)*(-1.0);
	memset( SR_order, -1, sizeof(int)*Wiener_N2 );
	Mat	cBlock = Mat::zeros(Size(Wiener_N1, Wiener_N1), CV_64FC1);
	Mat ser_Block = Mat::zeros( Wiener_N1, Wiener_N1, CV_64FC1);
    est_y( Range(index_cur.y, index_cur.y+Wiener_N1), Range(index_cur.x, index_cur.x+Wiener_N1)).copyTo(cBlock);

	int index_y, index_x;
	double distance = 0.0;
	for( int i=0; i<search_y-2*N1_half; i++)
		for (int j=0; j<search_x-2*N1_half; j++)
		{
			index_y = i+i_min;
			index_x = j+j_min;
			est_y( Range(index_y, index_y+(2*N1_half)+1), Range(index_x, index_x+(2*N1_half)+1)).copyTo(ser_Block);
			distance = Distance_wiener( cBlock, ser_Block );
			if ( distance < tau_match_wiener )
				SxR.at<double>(index_y, index_x) = distance;
		}

	vector<Vec2d> myvector;
	double coeff=0.0, pos=0.0;
	for (int i=0; i<height; i++)
		for (int j=0; j<width; j++)
		{
			coeff = SxR.at<double>(i,j);
			pos = (double)(i*width + j);
			if ( coeff >= 0)
				myvector.push_back( Vec2d(coeff, pos) );
		}
	sort ( myvector.begin(), myvector.end(), myobject );
	
	int foundnum = myvector.size();
	foundnum = (foundnum<Wiener_N2) ? foundnum : Wiener_N2;
	int found_pos=0, pos_x=0, pos_y=0;
	for (int k=0; k<foundnum; k++)
	{
		Vec2d it = myvector.at(k);
		found_pos = (int)( it[1] );
		pos_y = found_pos/width;
		pos_x = found_pos%width;
		est_y( Range(pos_y,pos_y+Wiener_N1), Range(pos_x, pos_x+Wiener_N1)).copyTo(Group_est[k]);
		noisy_y( Range(pos_y,pos_y+Wiener_N1), Range(pos_x, pos_x+Wiener_N1)).copyTo(Group_origin[k]);
		SR_order[k] = found_pos;
	}

	SxR.release();
	ser_Block.release();
	cBlock.release();
	myvector.clear();
	return foundnum;
}
/**********************************************
* Wiener filter denoising within a 3D group 
* group_est: 2D spatial version of similar blocks
             from the estimated image
* group_origin: 2D spatial version of similar blocks
             from the noisy image
* totalnum: number of similar blocks
* group: output inversed blocks of the group
* return the weight for next aggregation
***********************************************/
double BlockMatching::Denoising_wiener ( Mat* group_est, Mat* group_origin, int totalnum, Mat *&group)
{
	const double scale = 1./sqrt( (double)Wiener_N1*Wiener_N1*totalnum );
	double weight_xR = 0.0;

	Mat* est_trans = new Mat[totalnum];
	Mat* origin_trans = new Mat[totalnum];
	Mat cBlock = Mat::zeros(Wiener_N1, Wiener_N1, CV_64FC1);
	Mat tBlock = Mat::zeros(Wiener_N1, Wiener_N1, CV_64FC2);
	Mat time_in = Mat::zeros(totalnum, 1, CV_64FC2);
	Mat time_trans = Mat::zeros(totalnum, 1, CV_64FC2);
	Mat time_in_origin = Mat::zeros(totalnum, 1, CV_64FC2);
	Mat time_trans_origin = Mat::zeros(totalnum, 1, CV_64FC2);
	for (int k=0; k<totalnum; k++)
	{
		group_est[k].copyTo(cBlock);
		dft(cBlock, tBlock, DFT_COMPLEX_OUTPUT, 0);
		TransformPadding( tBlock, Wiener_N1 );
		tBlock.copyTo( est_trans[k] );
 
		group_origin[k].copyTo(cBlock);
		dft(cBlock, tBlock, DFT_COMPLEX_OUTPUT, 0);
		TransformPadding( tBlock, Wiener_N1 );
		tBlock.copyTo( origin_trans[k] );

		group[k].zeros( Wiener_N1, Wiener_N1, CV_64FC1);
	}

	double temp1 = 0.0, temp2 = 0.0 ;
	double temp3_sum = 0.0;
	double w_SxR = 0.0;
	double pix_x = 0.0, pix_y = 0.0;
	for ( int i=0; i<Wiener_N1; i++)
		for ( int j=0; j<Wiener_N1; j++)
		{
			for (int k=0; k<totalnum; k++)
			{
				time_in.at<Vec2d>(k,0) = est_trans[k].at<Vec2d>(i,j);
				time_in_origin.at<Vec2d>(k,0) = origin_trans[k].at<Vec2d>(i,j);
			}

			dft( time_in, time_trans, DFT_COMPLEX_OUTPUT, 0 );
			dft( time_in_origin, time_trans_origin, DFT_COMPLEX_OUTPUT, 0);

			for (int k=0; k<totalnum; k++)
			{
				Vec2d Pix = time_trans.at<Vec2d>(k,0);
				Vec2d Pix_origin = time_trans_origin.at<Vec2d>(k,0);

				pix_x = Pix[0] * scale;
				pix_y = Pix[1] * scale;
				temp1 = pix_x*pix_x+pix_y*pix_y;
				w_SxR = temp1/(temp1+gau_sigma*gau_sigma);
				temp2 = w_SxR*w_SxR;

				temp3_sum += temp2;

				pix_x = Pix_origin[0]*scale*w_SxR;
				pix_y = Pix_origin[1]*scale*w_SxR;

				time_trans_origin.at<Vec2d>(k,0) = Vec2d( pix_x, pix_y );
			}
			
			dft( time_trans_origin, time_in_origin, DFT_INVERSE + DFT_COMPLEX_OUTPUT + DFT_SCALE, 0 );
			for (int k=0; k<totalnum; k++)
			{
				origin_trans[k].at<Vec2d>(i,j) = time_in_origin.at<Vec2d>(k,0);
			}
		}

		for ( int k=0; k<totalnum; k++)
		{
			Mat inver_coeff = origin_trans[k];
			dft( inver_coeff, group[k], DFT_INVERSE + DFT_REAL_OUTPUT + DFT_SCALE, 0 );
			group[k] = group[k]/scale;
		}

	weight_xR = 1.0/temp3_sum;

    for (int k=0; k<totalnum; k++)
	{
		est_trans[k].release();
		origin_trans[k].release();
	}
	delete [] est_trans;
	delete [] origin_trans;
	cBlock.release();
	tBlock.release();
	time_in.release();
	time_trans.release();
	time_in_origin.release();
	time_trans_origin.release();
	return weight_xR;
}
/**********************************************
* Aggregation among the inversed denoised block back
  to the original position in the second step
* totalnum: number of similar blocks
* weight_xR: weight for the aggregation
* group: input inversed blocks of the group
* SR_order: input 1D MxN array with order index
***********************************************/
void BlockMatching::Aggregation_wiener( Mat& ebuff, Mat& wbuff, int totalnum, double weight_xR, Mat *group, int *SR_order)
{
	Mat W_xR = Mat::zeros( Wiener_N1, Wiener_N1, CV_64FC1);
	Wiener_Wwin2D.copyTo(W_xR);
	W_xR = W_xR * weight_xR;

	Mat temp1 = Mat::zeros( Wiener_N1, Wiener_N1, CV_64FC1);
	int found_pos = 0, found_i=0, found_j=0;
	for ( int k=0; k<totalnum; k++)
	{
		temp1 = group[k].mul(W_xR);
		found_pos = SR_order[k];
		found_i = found_pos/width;
		found_j = found_pos%width;
		for ( int i=0; i<Wiener_N1; i++)
		{
			double* Mi = temp1.ptr<double>(i);
			double* Ni = ebuff.ptr<double>(i+found_i);
			for ( int j=0; j<Wiener_N1; j++)
				Ni[j+found_j] += Mi[j];
		}
		for ( int i=0; i<Wiener_N1; i++)
		{
			double* Mi = W_xR.ptr<double>(i);
			double* Ni = wbuff.ptr<double>(i+found_i);
			for ( int j=0; j<Wiener_N1; j++)
				Ni[j+found_j] += Mi[j];
		}
	}

	W_xR.release();
	temp1.release();
}