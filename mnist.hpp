#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "Eigen/Core"

namespace mtk{
	class MNISTAnalizer{
		// MNIST のデータ変数
		const static int data_amount = 60000;
		const static int data_dim = 28;

		//乱数関係
		std::mt19937 mt19937;
		std::uniform_int_distribution<int> dist;

		//データ格納関係
		class MNISTData{
		public:
			float data[data_dim*data_dim];
			int label;
		};
		std::vector<MNISTData*> data_vector;
		int reverse(int n);
	public:
		MNISTAnalizer();
		~MNISTAnalizer();
		void setToMatrix(Eigen::MatrixXf& input,Eigen::MatrixXf& teacher,int batch_size);
		int loadMNISTData(std::string image_filename,std::string label_filename);
	};
}
