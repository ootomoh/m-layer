#include "layer.hpp"
#include <iostream>
#include <cmath>


const int input_size = 4;
const int layer0_output_size = 8;
const int layer1_output_size = 1;
const int batch_size = 10;
const int calc = 1000;

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
	for(int b = 0; b < batch_size;b++){

	}
}

int main(){
	Eigen::MatrixXf batch_input = Eigen::MatrixXf::Random(input_size,batch_size);
	Eigen::MatrixXf batch_teacher = Eigen::MatrixXf::Random(layer1_output_size,batch_size);
	Layer<Sigmoid,dSigmoid> layer0(input_size,layer0_output_size,batch_size);
	Layer<Sigmoid,dSigmoid> layer1(layer0_output_size,layer1_output_size,batch_size);
	for(int c = 0;c < calc;c++){
		initLearningDataset(batch_input,batch_teacher);
		auto layer0_out = layer0.forwardPropagate(batch_input);
		auto layer1_out = layer1.forwardPropagate(layer0_out);
		auto error = layer1_out - batch_teacher;
		std::cout<<"out="<<layer1_out<<std::endl;
		std::cout<<"err="<<error<<std::endl;
	}
}
