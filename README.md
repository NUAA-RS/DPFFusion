# ![](./topic.png)

![](./DPFFusion.png)

## (a)DMRM "![](./dmrm1.png)"

## (b)APSFM "![](./apsfm1.png)"

## (c)CDFM "![](./cdfm1.png)"


## Environments

```
python 3.10
cuda 11.8
```
### 1. Docker condition
- Docker (https://docs.docker.com/get-docker/)
- NVIDIA Docker(https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
### 2.  Docker mirror
```bash
docker build -t dpffusion:v1 .
```

## Install

```
conda create -n DPFFusion python=3.10
conda activate DPFFusion
pip install -r requirements.txt
```

## Train

The training process needs wandb API key.
The config file is `./configs/cfg.yaml`

```
python train.py
```

## Inference

```
python fuse.py
```

## Visualization "![](./visualizations/complete_quantitative_analysis.png)"

## Dataset

Datasets (M3FD, MSRS, RoadScene) are used to train. You can get it from [here](https://pan.baidu.com/s/1lmLmkbbSMr_PEwZdr3HPhg?pwd=ktvq).
