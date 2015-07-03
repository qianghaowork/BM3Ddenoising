#ifndef _RELATEDOPERATIONS_H

#define _RELATEDOPERATIONS_H

#include <cmath>
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"

using namespace std;
using namespace cv;
class RelatedOperations
{
private:

public:
	RelatedOperations();
	virtual ~RelatedOperations();
	double Io_Kaiser(double alpha);
	void Kaiser_Window(Mat &in_fit, int Nf, double beta);
};

#endif