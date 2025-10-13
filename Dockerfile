# 基础镜像：NVIDIA CUDA 11.6 + cuDNN 8.4.1 + Ubuntu 20.04
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

# 设置非交互模式，避免安装过程中的交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 设置Python3.9为默认Python版本
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 升级pip
RUN pip install --upgrade pip

# 安装PyTorch及相关依赖（匹配CUDA 11.6版本）
RUN pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 安装其他核心依赖
RUN pip install numpy==1.23.5 \
    opencv-python==4.6.0.66 \
    tqdm==4.64.1 \
    pillow==9.2.0 \
    scipy==1.9.3 \
    tensorboard==2.11.0

# 创建工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . /app

# 确保模型目录存在并复制预训练权重
RUN mkdir -p /app/models
COPY ./models/model-1_1_10_1.pth /app/models/

# 设置环境变量（可选，根据项目需求）
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# 容器启动时默认执行的命令（可根据需要修改）
CMD ["bash"]
