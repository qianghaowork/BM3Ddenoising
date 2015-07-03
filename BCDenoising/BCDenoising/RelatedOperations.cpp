#include "stdafx.h"
#include "RelatedOperations.h"

#define EPS 1.0e-8
#define MAXTERM 500

RelatedOperations::RelatedOperations()
{
}

RelatedOperations::~RelatedOperations()
{
}
/**********************************************
* IO function with parameter alpha
***********************************************/
double RelatedOperations::Io_Kaiser(double alpha)
{
	double  J = 1.0, K = alpha / 2.0, iOld = 1.0,iNew;
    bool    converge = false;

    // Use series expansion definition of Bessel.
    for( int i=1; i<MAXTERM; ++i )
    {
        J *= K/i;
        iNew = iOld + J*J;

        if( (iNew-iOld) < EPS )
        {
            converge = true;
            break;
        }
        iOld = iNew;
    }

    if( !converge )
        return 0;

    return iNew;
}
/**********************************************
* Generate the NfxNf Kaiser Window with parameter beta
***********************************************/
void RelatedOperations::Kaiser_Window(Mat &in_fit, int Nf, double beta)
{
	Mat Win1(Size(1,Nf), CV_64FC1);
	double bes = 1.0/Io_Kaiser(beta);
	double alpha, pos;

	double* Win1_data = Win1.ptr<double>(0);
	for (int i=0;i<(Nf+1)/2;++i)
	{
		pos = sqrt(double(i*(Nf-i-1.0)));
		alpha = 2*beta*pos/(Nf-1.0);
		Win1_data[i] = Io_Kaiser(alpha)*bes;
		Win1_data[Nf-1-i]  = Win1_data[i];
	}
	in_fit = Win1 * Win1.t();
}