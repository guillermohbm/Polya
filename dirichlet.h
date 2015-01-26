/*
*
	Polya library
	version c++ 11
*
*/
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

namespace Util {

	double lnchoose(int  n, int m){
		
		double nf=lgamma(n+1);
		double mf=lgamma(m+1);
		double nmmnf = lgamma(n-m+1);
		return (nf-(mf+nmmnf));
	
	}

	double bhattarchaya(VectorXd m1, VectorXd m2){
		
		ArrayXd r3;
		r3=m1.array()*m2.array();
		return sqrt(1-r3.sqrt().sum());
	
	}
	void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
	{
	
	    unsigned int numRows = matrix.rows()-1;
	    unsigned int numCols = matrix.cols();

	    if( rowToRemove < numRows )
	        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

	    matrix.conservativeResize(numRows,numCols);
	    
	}

	void flat(MatrixXd& mat){
	
		MatrixXd aux(1,mat.rows()*mat.cols());
		int z=0;
		for(int i=0;i<mat.rows();i++){
			for(int j=0;j<mat.cols();j++){
				aux(0,z)=mat(i,j);z++;
			}
		}
		mat.resize(aux.cols(),1);
		
	}
	MatrixXd average(MatrixXd a,MatrixXd weigths, int axis){
		MatrixXd r = a;
		Util::flat(weigths);
 		
		if(axis==0){

			for(int i=0;i<r.cols();i++){
				r.col(i)= r.col(i).array()*weigths.array();	
			}
			return (1/weigths.sum())*r.colwise().sum();	
		
		}else if(axis==1){

			for(int i=0;i<r.rows();i++){
				r.row(i)= r.row(i).array()*weigths.transpose().array();	
			}
			return (1/weigths.sum())*r.rowwise().sum().transpose();
		
		}else{
			cout << "invalid argument on average function AVERAGE" << endl;
			exit(EXIT_FAILURE);
		}
		
	}
	double median(MatrixXd med){
		MatrixXd aux = med;
		flat(aux);
		int n = aux.rows();
		if(n%2==0){
			return (aux(n/2,0)+aux(n/2-1,0))/2.0;

		}else{
			return aux(n/2,0);
		}
	}
	// Utils for digamma from http://fastapprox.googlecode.com/svn/trunk/fastapprox/src/fastonebigheader.h
	
	static inline float fastlog2 (float x)
	{
	  union { float f; uint32_t i; } vx = { x };
	  union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
	  float y = vx.i;
	  y *= 1.1920928955078125e-7f;

	  return y - 124.22551499f
	           - 1.498030302f * mx.f 
	           - 1.72587999f / (0.3520887068f + mx.f);
	}

	static inline float fastlog (float x)
	{
	  return 0.69314718f * fastlog2 (x);
	}


	static inline float fastdigamma (float x)
	{
	  float twopx = 2.0f + x;
	  float logterm = fastlog (twopx);

	  return (-48.0f + x * (-157.0f + x * (-127.0f - 30.0f * x))) /
	         (12.0f * x * (1.0f + x) * twopx * twopx)
	         + logterm;
	}

	MatrixXd psi(MatrixXd mat){
		MatrixXd res(mat.rows(),mat.cols());

		for(int i=0;i<mat.rows();i++){
			for(int j=0;j<mat.cols();j++){
				res(i,j)=fastdigamma(mat(i,j));
			}
		}
		return res;
	}
	float psi(float x){
		
		return fastdigamma(x);
	}

}

/*** CLASS POYLA ***/

class Polya{
	public:
		//constructors
		Polya(VectorXd a);
		Polya();
		//getters
		VectorXd getAlpha(){return alpha;};
		VectorXd getM(){return m;};
		double getS(){return s;};
		//setters
		void setAlpha(VectorXd a);
		//methods
		void meanprecision();
		MatrixXd dirichlet_moment_match(MatrixXd proportions, MatrixXd weigths);
		MatrixXd polya_moment_match(MatrixXd counts);
		MatrixXd fit_fixedPoint(MatrixXd counts,int maxIter,double tol); 
	private:
		VectorXd alpha;
		VectorXd m;
		double s;
		
};
Polya::Polya(){
	 alpha=VectorXd();
	 m=VectorXd();
	 s=0;
}
Polya::Polya(VectorXd a)
{
	alpha=a;
	m= VectorXd(alpha.size());
	s=0;
}
void Polya::setAlpha(VectorXd a){
	alpha=a;
	m= VectorXd(alpha.size());
	s=0;
}
void Polya::meanprecision(){
	s= alpha.sum();
	m= (1/s)*alpha;
}

MatrixXd Polya::dirichlet_moment_match(MatrixXd proportions, MatrixXd weigths){
	MatrixXd a;
	MatrixXd m2;
	MatrixXd aok,m2ok;
	double res=0;
	a= Util::average(proportions,weigths,0);
	m2 = Util::average(proportions.array()*proportions.array(),weigths,0);

	aok = a.transpose();
	m2ok = m2.transpose();

	for(int i=0;i<aok.rows();i++){
		if(aok(i,0)<=0){
			Util::removeRow(aok,i);
			Util::removeRow(m2ok,i);
			i--;
		}
	}
	res=Util::median((aok.array() - m2ok.array()) / (m2ok.array() - aok.array() * aok.array()));
	return a*res;

}

MatrixXd Polya::polya_moment_match(MatrixXd counts){
	MatrixXd norm_sum = counts.rowwise().sum();
	
	for(int i=0;i<counts.rows();i++){
		counts.row(i)= counts.row(i)*(1/norm_sum(i,0));
	}
	return dirichlet_moment_match(counts,norm_sum);
}

MatrixXd Polya::fit_fixedPoint(MatrixXd counts,int maxIter,double tol){ //incomplete
	int train=counts.rows();
	int D=counts.cols();
	int iter=0;
	MatrixXd alp, old_alp ;
	MatrixXd c;
	VectorXd d;
	double change = 2*tol;
    //counts = counts[sum(counts.A, axis=1) > 0, :]
    MatrixXd auxCounts = counts.rowwise().sum();
	for(int i=0;i<auxCounts.rows();i++){
		 if(auxCounts(i,0)<0){
		 	Util::removeRow(auxCounts,i);
		 	Util::removeRow(counts,i);
		 	i--;
		 }
	}

	//alpha = array(polya_moment_match(counts)).flatten()
 	alp = polya_moment_match(counts);
	c = MatrixXd::Zero(train,D);
	d = VectorXd::Zero(train);

	while(change > tol && iter < maxIter)
	{
		old_alp=alp;
		for (int i=0;i<train;i++){
			c.row(i)=Util::psi(counts.row(i)+alp)-Util::psi(alp);
			d[i]= Util::psi(counts.row(i).sum()+alp.sum())-Util::psi(alp.sum());
		}
		auxCounts=(c.colwise().sum())*(1.0/d.sum());
		alp=alp.array()*auxCounts.array();
		change = (alp-old_alp).array().abs().maxCoeff();
		iter++;
	}
	return alp;
	
}
