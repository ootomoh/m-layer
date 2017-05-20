#pragma once
#include "Eigen/Dense"

template<class ActivateFunc,class dActivateFunc>
class Layer{
	int input_size,output_size;
	int batch_size;
	Eigen::MatrixXf  w1;
	Eigen::MatrixXf  dw1;
	Eigen::MatrixXf  rdw1; // round d
	Eigen::MatrixXf  b1;
	Eigen::MatrixXf  db1;
	Eigen::MatrixXf  rdb1; // round d
	Eigen::MatrixXf  u1;
	Eigen::MatrixXf  z0;
	Eigen::MatrixXf  d1;
public:
	Layer(int input_size,int output_size,int batch_size):
		input_size(input_size),output_size(output_size),batch_size(batch_size)
	{
		z0 = Eigen::MatrixXf(input_size,batch_size);
		w1 = Eigen::MatrixXf::Random(output_size,input_size);
		dw1 = Eigen::MatrixXf::Random(output_size,input_size);
		rdw1 = Eigen::MatrixXf::Random(output_size,input_size);
		b1 = Eigen::MatrixXf::Random(output_size,1);
		db1 = Eigen::MatrixXf::Random(output_size,1);
		rdb1 = Eigen::MatrixXf::Random(output_size,1);
		u1 = Eigen::MatrixXf(output_size,batch_size);
	}
	Eigen::MatrixXf forwardPropagate(const Eigen::MatrixXf& input){
		z0 = input;
		u1 = w1 * z0 + b1 * Eigen::MatrixXf::Constant(1,batch_size,1.0f);
		return  u1.unaryExpr(ActivateFunc());
	}
	Eigen::MatrixXf backPropagate(const Eigen::MatrixXf& d2,const Eigen::MatrixXf& w2){
		d1 = u1.unaryExpr( dActivateFunc() ).array() * (w2.transpose()*d2).array();
		rdw1 = d1*z0.transpose()/static_cast<float>(batch_size);
		rdb1 = d1*Eigen::MatrixXf::Constant(batch_size,1,1.0f)/static_cast<float>(batch_size);
		return d1;
	}
	void reflect(){
		const float lerning_rate = 0.03f;
		dw1 = rdw1 * (-lerning_rate);
		db1 = rdb1 * (-lerning_rate);
		w1 = w1 + dw1;
		b1 = b1 + db1;
	}
	void setD2(const Eigen::MatrixXf& d2){
		d1 = d2;
	}
	Eigen::MatrixXf getW() const{
		return w1;
	}
};
