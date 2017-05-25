exec_main: main.cpp layer.hpp
	g++ -std=c++11 main.cpp -o $@ 

clean:
	rm exec*

run: exec_main
	time ./exec_main

errordata: exec_main
	make run | grep 'e=' | sed -e 's/e=//g' -e 's/\s/,/g' -e 's/,,/,/g' -e 's/^,//g' -e 's/-//g' > error-rate.csv 
