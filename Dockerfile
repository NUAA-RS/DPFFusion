# 基础镜像：NVIDIA CUDA 11.6 + cuDNN 8.4.1（兼容PyTorch 1.12.0）
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

# 设置非交互模式，避免安装过程中弹出配置窗口
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 设置Python默认版本
RUN ln -s /usr/bin/python3 /usr/bin/python

# 安装核心依赖项（按用户指定版本）
RUN pip3 install --no-cache-dir \
    torch==1.12.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html \
    torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html \
    numpy==1.23.5 \
    opencv-python==4.6.0.66 \
    tqdm==4.64.1 \
    kornia==0.7.3 \
    matplotlib==3.7.0 \
    Pillow==10.0.1 \
    PyWML==6.0.1 \
    scikit_learn==1.3.0 \
    sclpy==1.14.0 \
    thopen==1.1.post2290072288 \
    wandb==0.17.4

# 创建工作目录
WORKDIR /app

# 复制仓库中的所有文件到容器（需在.gitignore中排除不必要文件）
COPY . /app

# 声明容器启动时的默认命令（可选，可留空）

CMD ["bash"]
