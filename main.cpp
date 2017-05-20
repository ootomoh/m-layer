#include "layer.hpp"
#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <random>


#define SHOW_INPUT
#define SHOW_OUTPUT

const int input_size = 4;
const int layer0_output_size = 8;
const int layer1_output_size = 1;
const int batch_size = 10;
const int calc = 1;

class Sigmoid{
public:
	float operator()(const float x) const{
		return 1.0f/(1.0f+std::exp(-x));
	}
};
class dSigmoid{
public:
	float operator()(const float x) const{
		float s = Sigmoid()(x);
		return s * (1.0f-s);
	}
};

void initLearningDataset(Eigen::MatrixXf &batch_input,Eigen::MatrixXf &batch_teacher){
	std::mt19937 mt(std::random_device{}());
	std::uniform_int_distribution<int> dist(0,1);
	float input[input_size];
	for(int b = 0; b < batch_size;b++){
		float sum = 0.0f;
		float *ptr = batch_input.data()+sizeof(float)*b;
		std::generate(ptr,ptr+input_size,[&mt,&dist,&sum](){return (dist(mt)==0?0.0f:(sum+=1.0f,1.0f));});
		if( sum > input_size/2.0f-1.0f)
			batch_teacher(0,b)=1.0f;
		else
			batch_teacher(0,b)=0.0f;
	}
}

int main(){
	Eigen::MatrixXf batch_input = Eigen::MatrixXf::Random(input_size,batch_size);
	Eigen::MatrixXf batch_teacher = Eigen::MatrixXf::Random(layer1_output_size,batch_size);
	Layer<Sigmoid,dSigmoid> layer0(input_size,layer0_output_size,batch_size);
	Layer<Sigmoid,dSigmoid> layer1(layer0_output_size,layer1_output_size,batch_size);
	for(int c = 0;c < calc;c++){
		initLearningDataset(batch_input,batch_teacher);
#ifdef SHOW_INPUT
		std::cout<<"input="<<std::endl<<batch_input<<std::endl;
		std::cout<<"teacher="<<std::endl<<batch_teacher<<std::endl;
#endif
		auto layer0_out = layer0.forwardPropagate(batch_input);
		auto layer1_out = layer1.forwardPropagate(layer0_out);
		auto error = layer1_out - batch_teacher;
#ifdef SHOW_OUTPUT
		std::cout<<"out="<<layer1_out<<std::endl;
		std::cout<<"err="<<error<<std::endl;
#endif
	}
}
