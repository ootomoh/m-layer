#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <random>


//#define SHOW_INPUT
#define SHOW_OUTPUT
//#define SHOW_WEIGHT
#define SHOW_WEIGHT_WHEN_DESTROY

#include "layer.hpp"

const int input_size = 8;
const int layer0_output_size = 32;
const int layer1_output_size = 1;
const int batch_size = 32;
const int calc = 200000;

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
class Step{
public:
	float operator()(const float x) const{
		return (x>0.0f?1.0f:0.0f);
	}
};
class ReLU{
public:
	float operator()(const float x) const{
		return (x>0.0f?x:0.0f);
	}
};

void initLearningDataset(Eigen::MatrixXf &batch_input,Eigen::MatrixXf &batch_teacher){
	std::mt19937 mt(std::random_device{}());
	std::uniform_int_distribution<int> dist(0,1);
	float input[input_size];
	for(int b = 0; b < batch_size;b++){
		float sum = 0.0f;
		float *ptr = batch_input.data()+b*input_size;
		std::generate(ptr,ptr+input_size,[&mt,&dist,&sum](){return (dist(mt)==0?0.0f:(sum+=1.0f,1.0f));});
		if( sum < input_size/4.0f || 3.0f*input_size/4.0f < sum )
		//if( (sum/2.0f)-static_cast<float>(static_cast<int>(sum/2.0f)) > 0.49f )
			batch_teacher(0,b)=0.0f;
		else
			batch_teacher(0,b)=1.0f;
	}
}

int main(){
	Eigen::MatrixXf batch_input = Eigen::MatrixXf::Random(input_size,batch_size);
	Eigen::MatrixXf batch_teacher = Eigen::MatrixXf::Random(layer1_output_size,batch_size);
	Layer<Sigmoid,dSigmoid> layer0(input_size,layer0_output_size,batch_size,"layer0");
	Layer<Step,dSigmoid> layer1(layer0_output_size,layer1_output_size,batch_size,"layer1");
	for(int c = 0;c < calc;c++){
		initLearningDataset(batch_input,batch_teacher);
#ifdef SHOW_INPUT
		std::cout<<"i="<<batch_input<<std::endl;
		std::cout<<"t="<<batch_teacher<<std::endl;
#endif
		auto layer0_out = layer0.forwardPropagate(batch_input);
		auto layer1_out = layer1.forwardPropagate(layer0_out);
		auto error = layer1_out - batch_teacher ;
#ifdef SHOW_OUTPUT
		std::cout<<"o="<<layer1_out<<std::endl;
		std::cout<<"e="<<error<<std::endl;
#endif
		layer1.setD2(error);
		layer0.backPropagate(error,layer1.getW());

		layer1.reflect();
		layer0.reflect();
	}
#ifdef SHOW_WEIGHT_WHEN_DESTROY
	layer0.showWeight();
	layer1.showWeight();
#endif
}
