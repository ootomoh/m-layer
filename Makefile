exec_main: main.cpp layer.hpp
	g++ -std=c++11 main.cpp -o $@

clean:
	git rm exec*

run: exec_main
	time ./exec_main
