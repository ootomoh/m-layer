#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <stdio.h>
#include "Eigen/Core"

namespace mtk{
	class MNISTLoader{
		// MNIST のデータ変数
		const static int train_data_amount = 60000;
		const static int data_dim = 28;

		//乱数関係
		std::mt19937 mt19937;
		std::uniform_int_distribution<int> dist;

		//データ格納関係
		class MNISTData{
		public:
			float data[data_dim*data_dim];
			int label;
			void print(){
				for(int i = 0;i < data_dim;i++){
					for(int j = 0;j , data_dim;j++){
						printf("%.2lx",((int)data[j+i*data_dim])&0xff);
					}
				}
			}
		};
		std::vector<MNISTData*> train_data_vector;
		std::vector<MNISTData*> test_data_vector;
		int reverse(int n);
		int loadMNISTData(std::string image_filename,std::string label_filename,std::vector<MNISTData*> &data_vector);
	public:
		MNISTLoader();
		~MNISTLoader();
		void setTrainDataToMatrix(Eigen::MatrixXf& input,Eigen::MatrixXf& teacher,int batch_size);
		int setTestDataToMatrix(Eigen::MatrixXf& input,int index);
		int loadMNISTTrainData(std::string image_filename,std::string label_filename);
		int loadMNISTTestData(std::string image_filename,std::string label_filename);
		void showMNISTTestData(int index);
	};
}
