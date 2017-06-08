#include "../mnist.hpp"

int main(){
	const int batch_size = 10;
	const int input_size = 28*28;
	const int teacher_size = 10;
	const int loop = 1100000;
	Eigen::MatrixXf input(input_size,batch_size);
	Eigen::MatrixXf teacher(teacher_size,batch_size);

	mtk::MNISTAnalizer mnist;
	if(mnist.loadMNISTData("../train-images-idx3-ubyte","../train-labels-idx1-ubyte")){
		std::cerr<<"Loading error"<<std::endl;
		return 1;
	}

	for(int i = 0;i < loop;i++){
		std::cout<<"loop : "<<i<<" / "<<loop<<std::endl;
		mnist.setToMatrix(input,teacher,batch_size);
	}

	std::cout<<"input"<<std::endl<<input<<std::endl;
	std::cout<<"teacher"<<std::endl<<teacher<<std::endl;
}
