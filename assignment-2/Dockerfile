FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y gcc g++ build-essential && \
    apt-get clean

WORKDIR /app

COPY attention.cpp .
COPY input.txt .

RUN gcc attention.cpp -o attention -lstdc++ -lpthread

# 쉘을 통해 명령어 전체 실행 (리디렉션 포함)
CMD ["/bin/sh", "-c", "./attention 2 < input.txt"]