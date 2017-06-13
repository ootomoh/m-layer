#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <random>
#include <ctime>
#include <chrono>

#include "mnist.hpp"


//#define SHOW_INPUT
//#define SHOW_OUTPUT
//#define SHOW_WEIGHT
//#define SHOW_WEIGHT_WHEN_DESTROY
//#define SHOW_ERROR

#define TEST

#include "layer.hpp"

const int input_size = 28*28;
const int layer0_output_size = 12*28;
const int layer1_output_size = 10;
const int batch_size = 1<<13;
const int calc = 1200;

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


int main(){
	std::cout<<">>>default.out"<<std::endl;
	std::srand(std::time(NULL));
	Eigen::MatrixXf batch_input = Eigen::MatrixXf::Random(input_size,batch_size);
	Eigen::MatrixXf batch_teacher = Eigen::MatrixXf::Random(layer1_output_size,batch_size);
	HiddenLayer<Sigmoid,dSigmoid> layer0(input_size,layer0_output_size,batch_size,"layer0");
	SoftmaxLayer layer1(layer0_output_size,layer1_output_size,batch_size,"layer1");
	mtk::MNISTLoader mnist;
	if(mnist.loadMNISTTrainData("./train-images-idx3-ubyte","./train-labels-idx1-ubyte")){
		std::cerr<<"Invalid training data"<<std::endl;
		return 1;
	}
#ifdef TEST
	if(mnist.loadMNISTTestData("./t10k-images-idx3-ubyte","./t10k-labels-idx1-ubyte")){
		std::cerr<<"Invalid test data"<<std::endl;
		return 1;
	}
#endif
	auto start_time = std::chrono::system_clock::now();
	for(int c = 0;c < calc;c++){
		std::cout<<"calc "<<c<<std::endl;
#if defined(SHOW_INPUT) || defined(SHOW_OUTPUT)
		std::cout<<"training : "<<c<<" / "<<calc<<std::endl;
#endif
		mnist.setTrainDataToMatrix(batch_input,batch_teacher,batch_size);
		//initLearningDataset(batch_input,batch_teacher);//
#ifdef SHOW_INPUT
		std::cout<<"i="<<std::endl<<batch_input<<std::endl;
		std::cout<<"t="<<std::endl<<batch_teacher<<std::endl;
#endif
		auto layer0_out = layer0.forwardPropagate(batch_input);
		auto layer1_out = layer1.forwardPropagate(layer0_out);
		//auto error = - layer1_out + batch_teacher ;
		auto error = layer1_out - batch_teacher ;
#ifdef SHOW_OUTPUT
		std::cout<<"o="<<layer1_out<<std::endl;
#endif
#ifdef SHOW_ERROR
		std::cout<<"e="<<error<<std::endl;
#endif
		layer1.backPropagate(error,layer1.getW());
		layer0.backPropagate(error,layer1.getW());

		layer1.reflect();
		layer0.reflect();

#ifdef TEST
	std::cout<<">>>correct_ratio.out"<<std::endl;
		if( c % 10 == 9){
			std::cout<<"----"<<(c+1)<<" test----"<<std::endl;
			int correct_count = 0;
			const int test_amount = 10000;
			Eigen::MatrixXf test_input(28*28,1);
			for(int j = 0;j < test_amount;j++){
				int correct = mnist.setTestDataToMatrix(test_input,j);
				//	std::cout<<"correct = "<<correct<<std::endl;
				auto layer0_out = layer0.testDataForwardPropagate(test_input);
				auto layer1_out = layer1.testDataForwardPropagate(layer0_out);
				if(layer1_out.maxCoeff() == layer1_out(correct,0)){
					correct_count++;
				}
			}
			std::cout<<"correct ratio : "<<correct_count/static_cast<float>(test_amount)*100.0f<<" %"<<std::endl; 
		}
	std::cout<<">>>default.out"<<std::endl;
#endif
	}
	auto stop_time = std::chrono::system_clock::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(stop_time-start_time).count();
	std::cout<<"elapsed time "<<elapsed_time<<" [s]"<<std::endl<<"calculation count"<<calc<<std::endl;
#ifdef SHOW_WEIGHT_WHEN_DESTROY
	std::cout<<">>>weight"<<std::endl;
	layer0.showWeight();
	layer1.showWeight();
#endif
#ifdef TEST
	std::cout<<"----last test----"<<std::endl;
	int correct_count = 0;
	const int test_amount = 10000;
	Eigen::MatrixXf test_input(28*28,1);
	for(int j = 0;j < test_amount;j++){
		int correct = mnist.setTestDataToMatrix(test_input,j);
		//	std::cout<<"correct = "<<correct<<std::endl;
		auto layer0_out = layer0.testDataForwardPropagate(test_input);
		auto layer1_out = layer1.testDataForwardPropagate(layer0_out);
		if(layer1_out.maxCoeff() == layer1_out(correct,0)){
			correct_count++;
		}
	}
	std::cout<<"correct ratio : "<<correct_count/static_cast<float>(test_amount)*100.0f<<" %"<<std::endl; 
#endif
}
