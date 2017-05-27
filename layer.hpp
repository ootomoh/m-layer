#pragma once
#include <string>
#include <stdio.h>
#include "Eigen/Dense"

//#define USE_ADAGRAD
//#define USE_MOMENTUM

template<class ActivateFunc,class dActivateFunc>
class Layer{
	int input_size,output_size;
	int batch_size;
	std::string layer_name;
	Eigen::MatrixXf  w1;
	Eigen::MatrixXf  dw1;
	Eigen::MatrixXf  rdw1; // round d
	Eigen::MatrixXf  b1;
	Eigen::MatrixXf  db1;
	Eigen::MatrixXf  rdb1; // round d
	Eigen::MatrixXf  u1;
	Eigen::MatrixXf  z0;
	Eigen::MatrixXf  d1;
#ifdef USE_ADAGRAD
	Eigen::MatrixXf  adagrad_w1;
	Eigen::MatrixXf  adagrad_b1;
#endif
public:
	Layer(int input_size,int output_size,int batch_size,std::string layer_name=""):
		input_size(input_size),output_size(output_size),batch_size(batch_size),layer_name(layer_name)
	{
		z0 			= Eigen::MatrixXf(input_size,batch_size);
		w1 			= Eigen::MatrixXf::Random(output_size,input_size);
		dw1 		= Eigen::MatrixXf::Random(output_size,input_size);
		rdw1 		= Eigen::MatrixXf::Random(output_size,input_size);
		//b1 			= Eigen::MatrixXf::Constant(output_size,1,0.0f);
		b1 			= Eigen::MatrixXf::Random(output_size,1);
		db1 		= Eigen::MatrixXf::Random(output_size,1);
		rdb1 		= Eigen::MatrixXf::Random(output_size,1);
		u1 			= Eigen::MatrixXf::Zero(output_size,batch_size);
#ifdef USE_ADAGRAD
		adagrad_w1 	= Eigen::MatrixXf::Zero(output_size,input_size);
		adagrad_b1 	= Eigen::MatrixXf::Zero(output_size,1);
#endif
	}
	~Layer(){
	}
	Eigen::MatrixXf forwardPropagate(const Eigen::MatrixXf& input){
#ifdef SHOW_WEIGHT
		std::cout<<layer_name<<":w = "<<std::endl<<w1<<std::endl;
		std::cout<<layer_name<<":b = "<<std::endl<<b1<<std::endl;
#endif
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
#ifdef USE_MOMENTUM
		const float attenuation_rate = 0.9f;
#endif
#ifdef USE_ADAGRAD
		const float adagrad_epsilon = 1.0f;
		const float lerning_rate = 0.1f;
		auto adagrad_make = [&adagrad_epsilon](float x)->float{return 1.0f/(std::sqrt(x)+adagrad_epsilon);};
		auto adagrad_square = [](float x)->float{return x*x;};
		adagrad_w1 = adagrad_w1 + rdw1.unaryExpr(adagrad_square);
		adagrad_b1 = adagrad_b1 + rdb1.unaryExpr(adagrad_square);
#else
		const float lerning_rate = 0.01f;
#endif
		dw1 = rdw1.array()
#ifdef USE_ADAGRAD
			* adagrad_w1.unaryExpr(adagrad_make).array()
#endif
			* (-lerning_rate)
#ifdef USE_MOMENTUM
			+ dw1.array() * attenuation_rate
#endif
			;
		db1 = rdb1.array() 
#ifdef USE_ADAGRAD
			* adagrad_b1.unaryExpr(adagrad_make).array() 
#endif
			* (-lerning_rate)
#ifdef USE_MOMENTUM
			+ db1.array() * attenuation_rate
#endif
			;
		//db1 = rdb1 * (-lerning_rate) + db1 * attenuation_rate;
		std::cout<<layer_name<<":dw = "<<std::endl<<dw1<<std::endl;
		std::cout<<layer_name<<":db = "<<std::endl<<db1<<std::endl;

		w1 = w1 + dw1;
		b1 = b1 + db1;
	}
	void setD2(const Eigen::MatrixXf& d2){
		d1 = d2;
	}
	Eigen::MatrixXf getW() const{
		return w1;
	}
	void showWeight(){
		std::cout<<layer_name.c_str()<<":w = "<<std::endl<<w1<<std::endl;
		std::cout<<layer_name.c_str()<<":b = "<<std::endl<<b1<<std::endl;
	}
};
