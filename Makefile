exec_main: obj/main.o obj/mnist.o 
	g++ -std=c++11 $+ -o $@ -fopenmp

obj/%.o: %.cpp layer.hpp
	g++ -c -std=c++11 $< -o $@ -fopenmp


clean:
	rm exec*
	rm obj/*.o

cleanlog:
	rm -rf logfiles/*

run: exec_main
	nohup sh -c 'OMP_NUM_THREADS=4 /home/mutsuki/works/l-program/cpp/m-layer/exec_main | /home/mutsuki/works/l-program/cpp/m-layer/output-saver -d logfiles' &
