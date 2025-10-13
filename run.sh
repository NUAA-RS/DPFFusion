#!/bin/bash

# 训练命令示例
function train_model {
    docker run -it --gpus all \
        -v $(pwd)/datasets:/app/datasets \
        -v $(pwd)/results:/app/results \
        dpffusion:v1 \
        python train.py --epochs 200 --batch_size 16 --ckpt_save_path ./models
}

# 测试命令示例
function test_model {
    docker run -it --gpus all \
        -v $(pwd)/results:/app/results \
        dpffusion:v1 \
        python test.py --dataset M3FD --ckpt ./models/model-1_1_10_1.pth
}

# 显示帮助信息
function show_help {
    echo "用法: ./run.sh [command]"
    echo "命令选项:"
    echo "  train    - 启动训练"
    echo "  test     - 启动测试"
    echo "  help     - 显示帮助信息"
}

# 根据输入参数执行相应命令
case "$1" in
    train)
        train_model
        ;;
    test)
        test_model
        ;;
    help)
        show_help
        ;;
    *)
        echo "未知命令: $1"
        show_help
        exit 1
        ;;
esac