#include "layer.hpp"
#include <cmath>


class Sigmoid{
public:
	float operator()(float x){
		return 1.0f/(1.0f+std::exp(-x));
	}
};

int main(){
	Layer<Sigmoid> layer(10,10);
}
