CXX=g++

exec_main: obj/main.o obj/mnist.o 
	$(CXX)  $+ -o $@ -fopenmp

obj/%.o: %.cpp layer.hpp
	$(CXX) -c -std=c++11 $< -o $@ -fopenmp


clean:
	rm exec*
	rm obj/*.o

cleanlog:
	rm -rf logfiles/*

run: exec_main
	nohup sh -c './exec_main | ./output-saver -d logfiles' &
