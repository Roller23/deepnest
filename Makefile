CC=clang++
flags=-std=c++17 -O3
network=network/network.cpp
out=main

all:
	$(CC) -o ${out} main.cpp ${network} ${flags}

run:
	./${out}

debug:
	lldb ./${out}

clean:
	rm ./${out}