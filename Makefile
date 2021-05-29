CC=g++
flags=-std=c++17 -O3 -lm
files=example.cpp deepnest/network.cpp deepnest/utils.cpp
out=main

all:
	$(CC) -o ${out} ${files} ${flags}

run:
	./${out}