#include "Eigen/Dense"

template<class Func>
class Layer{
	int input_size,output_size;
	Eigen::MatrixXf w1;
	Eigen::VectorXf b1;
	Eigen::VectorXf u1;
	Eigen::VectorXf z0;
public:
	Layer(int input_size,int output_size):
		input_size(input_size),output_size(output_size)
	{
		z0 = Eigen::VectorXf(input_size);
		w1 = Eigen::MatrixXf::Random(output_size,input_size);
		b1 = Eigen::VectorXf::Random(output_size);
		u1 = Eigen::VectorXf(output_size);
	}
	Eigen::VectorXf forwardPropagate(Eigen::VectorXf input){
		z0 = input;
		u1 = w1 * z0 + b1;
		return u1.unaryExpr(Func());
	}
};
