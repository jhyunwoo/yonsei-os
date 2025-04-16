CXX = gcc
CXXFLAGS = -std=c++11 -Wall
LDFLAGS = -lstdc++

TARGETS = attention attention_mp multiHeadAttention

all: $(TARGETS)

attention: attention.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -lpthread

attention_mp: attention_mp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -lpthread

multiHeadAttention: multiHeadAttention.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)
