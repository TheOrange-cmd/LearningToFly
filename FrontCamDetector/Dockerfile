# Dockerfile for onnx2c
# This Dockerfile sets up an environment for converting ONNX models to C code using onnx2c.
FROM ubuntu:20.04

# Set noninteractive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    protobuf-compiler \
    libprotobuf-dev \
    libopenblas-dev

# Install Python dependencies
RUN pip3 install onnx==1.12.0 numpy

# Clone and prepare onnx2c
WORKDIR /app
RUN git clone https://github.com/kraiskil/onnx2c.git && \
    cd onnx2c && \
    git submodule update --init

# Find and fix the onnx.proto file
RUN cd onnx2c && \
    find . -name "onnx.proto" -exec sed -i 's/option optimize_for = LITE_RUNTIME;//' {} \;

# Fix main.cc for ParseFromIstream
RUN cd onnx2c && \
    sed -i 's/onnx_model.ParseFromIstream(&input);/std::string model_str((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()); onnx_model.ParseFromString(model_str);/' src/main.cc

# Build in Debug mode as recommended in the FAQ
RUN cd onnx2c && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make onnx2c

# Make the binary available
RUN cp /app/onnx2c/build/onnx2c /usr/local/bin/

# Set working directory for model conversion
WORKDIR /data

# Default command
ENTRYPOINT ["onnx2c"]
CMD ["--help"]