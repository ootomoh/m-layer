#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include "Eigen/Core"

namespace mtk{
	class MNISTAnalizer{
		class MNISTData{
			float data[28];
			int label;
		};
		int reverse(int n);
	public:
		MNISTAnalizer(){}
		int setToMatrix(Eigen::MatrixXf& input,int batch_size);
		int loadMNISTData(std::string image_filename,std::string label_filename);
	}
};
