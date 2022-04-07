FROM nvidia/cuda:11.6.1-devel-ubuntu18.04

WORKDIR /
COPY . .
RUN apt-get update && apt-get install -y cmake libgmp-dev && mkdir /build && cd /build && cmake .. && make

CMD [ "/build/main" ]
