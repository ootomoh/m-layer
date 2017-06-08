exec_main: obj/main.o obj/mnist.o
	g++ -std=c++11 $+ -o $@ 

obj/%.o: %.cpp
	g++ -c -std=c++11 $+ -o $@ 


clean:
	rm exec*

run: exec_main
	time ./exec_main

errordata: exec_main
	make run | grep 'e=' | sed -e 's/e=//g' -e 's/\s/,/g' -e 's/,,/,/g' -e 's/^,//g' -e 's/-//g' > error-rate.csv 
