# Makefile 예시
CXX = gcc
CXXFLAGS = -std=c++17 -Wall
TARGET = attention
SRC = attention.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) -lstdc++

clean:
	rm -f $(TARGET)