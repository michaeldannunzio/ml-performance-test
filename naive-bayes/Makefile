all :
	make run && make clean
	
clean :
		rm -f ./naive-bayes

build :
	clang++ -std=c++11 -stdlib=libc++ naive-bayes.cpp -o naive-bayes

exec :
	./main

run :
	make build && make exec
