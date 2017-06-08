#include "mnist.hpp"

using namespace mtk;

// コンストラクタでは乱数の初期化のみ行う
MNISTAnalizer::MNISTAnalizer(){
	std::random_device rnd;
	mt19937.seed(rnd());
	dist = std::uniform_int_distribution<int>{0,data_amount-1};
}
int MNISTAnalizer::reverse(int n){
	char a0,a1,a2,a3;
	a0 = (n>>24) & 255;
	a1 = (n>>16) & 255;
	a2 = (n>>8) & 255;
	a3 = n & 255;
	return ((int)a3 << 24) + ((int)a2 << 16) + ((int)a1 << 8) + a0;
}

void MNISTAnalizer::setToMatrix(Eigen::MatrixXf& input,Eigen::MatrixXf& teacher,int batch_size){
	teacher.setZero();
	for(int i = 0;i < batch_size;i++){
		int index = dist( mt19937 );
		MNISTData* data = data_vector[index];
		for(int j = 0;j < data_dim*data_dim;j++){
			input(j,i) = data->data[j];
		}
		teacher(data->label,i) = 1.0f;
	}
}

int MNISTAnalizer::loadMNISTData(std::string image_filename,std::string label_filename){
	std::ifstream image_ifs(image_filename,std::ios::binary);
	std::ifstream label_ifs(label_filename,std::ios::binary);
	if(! image_ifs|| !label_ifs ){
		return 1;
	}

	int8_t magic_number,amount,row,col;
	int label;
	image_ifs.read((char*)&magic_number,sizeof(magic_number));
	magic_number = reverse( magic_number );
	image_ifs.read((char*)&amount,sizeof(amount));
	amount = reverse( amount );
	image_ifs.read((char*)&row,sizeof(row));
	row = reverse( row );
	image_ifs.read((char*)&col,sizeof(col));
	col = reverse( col );
	label_ifs.read((char*)&magic_number,sizeof(magic_number));
	magic_number = reverse( magic_number );
	label_ifs.read((char*)&amount,sizeof(amount));
	amount = reverse( amount );
	for(int a = 0;a < data_amount;a++){
		MNISTData *data = new MNISTData;
		label_ifs.read((char*)&label,sizeof(char));
		data->label = reverse( label );
		for(int i = 0;i < 28*28;i++){
			image_ifs.read((char*)data->data + i,sizeof(char));
		}
		data_vector.push_back(data);
	}
	image_ifs.close();
	label_ifs.close();
}

MNISTAnalizer::~MNISTAnalizer(){
	for(auto data : data_vector){
		delete data;
	}
}
