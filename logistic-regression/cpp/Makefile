all :
	make run && make clean
	
clean :
		rm -f ./logistic-regression

build :
	clang++ -std=c++11 -stdlib=libc++ logistic-regression.cpp -o logistic-regression

exec :
	./logistic-regression

run :
	make build && make exec
