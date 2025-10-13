# ![](./topic.png)

![](./DPFFusion.png)

## (a)DMRM ![](./dmrm1.png)

## (b)APSFM ![](./apsfm1.png)

## (c)CDFM ![](./cdfm1.png)


## Environments

```
python 3.10
cuda 11.6
```
### Docker Usage Guide for DPFFusion

This guide provides step-by-step instructions to replicate the DPFFusion training/testing environment using Docker, ensuring cross-platform reproducibility.
1. Environment Requirements
Before starting, ensure your system meets the following prerequisites:

•Operating System: Windows 10/11 (WSL 2 enabled) or Linux (Ubuntu 18.04+/CentOS 7+)

•NVIDIA Hardware: GPU with CUDA support (compute capability ≥ 6.0, e.g., GTX 1080Ti, RTX 2080Ti, A100)

•NVIDIA Drivers:

◦Windows: Version ≥ 470.57.02 (minimum for CUDA 11.6)

◦Linux: Version ≥ 470.57.02 (install via `sudo apt install nvidia-driver-470` for Ubuntu)

•Docker Software:

◦Docker Desktop (Windows/Linux) ≥ 4.0.0 (with GPU support enabled)

◦For Windows: Ensure WSL 2 is installed (follow Microsoft's guide if missing)

3. Build the Docker Image
   
  2.1. Clone the DPFFusion repository to your local machine (skip if already cloned):

```
git clone https://github.com/NUAA-RS/DPFFusion.git
cd DPFFusion
```

  2.2. Ensure the pre-trained model `model-1_1_10_1.pth` is placed in the `./models` directory (as provided in your project).

  2.3. Build the Docker image using the `Dockerfile` (replace `dpffusion:v1` with your preferred image name/tag):

```
docker build -t dpffusion:v1 .
```

◦Note: The first build may take 20–30 minutes (depends on network speed) as it downloads the CUDA base image and Python dependencies.

◦Verify the image is built successfully:

```
docker images | grep dpffusion
```
You should see output like:

```
dpffusion   v1   abc123456789   10 minutes ago   12.5GB
```
3. Run the Docker Container
Use the following commands to start training or testing. The `-v` flag mounts local directories to the container for persistent storage (e.g., datasets, results).

  3.1. Training Command
Run training with custom parameters (adjust `--epochs`, `--batch_size`, and mount paths as needed):

```
docker run -it --gpus all \
  -v $(pwd)/datasets:/app/datasets \  # Mount local dataset dir to container
  -v $(pwd)/results:/app/results \    # Mount local results dir to save outputs
  -v $(pwd)/models:/app/models \      # Mount local model dir to save checkpoints
  dpffusion:v1 \
  python train.py \
    --epochs 200 \
    --batch_size 16 \
    --ckpt_save_path ./models \       # Save new checkpoints to mounted ./models
    --dataset M3FD                    # Replace with your target dataset
```
•Explanation:

◦`--gpus all`: Enables all available GPUs for the container.

◦`-v $(pwd)/datasets:/app/datasets`: Ensures the container uses your local dataset (avoids re-uploading data to the container).

◦Training logs and checkpoints will be saved to `./results` and `./models` (local directories, persistent after the container exits).

  3.2. Testing Command
Test the pre-trained model `model-1_1_10_1.pth` (adjust `--dataset` and model path as needed):

```
docker run -it --gpus all \
  -v $(pwd)/results:/app/results \    # Mount local dir to save test outputs
  dpffusion:v1 \
  python test.py \
    --dataset M3FD \                  # Replace with your test dataset
    --ckpt ./models/model-1_1_10_1.pth  # Path to pre-trained model (inside container)
```
•Verify Results: After testing, check the local `./results` directory for output images and evaluation metrics (e.g., MI, VIF scores).

4. Troubleshooting
   
•GPU Not Detected: Ensure `--gpus all` is included in the `docker run` command, and Docker Desktop GPU support is enabled (Windows: Settings → Resources → GPU; Linux: Install `nvidia-container-toolkit`).

•Slow Build Speed: Use a Docker mirror (e.g., Alibaba Cloud) by adding `RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list` to the `Dockerfile` before `apt-get update`.

•Permission Errors: On Linux, run `sudo chmod -R 755 ./datasets ./results ./models` to grant read/write permissions for mounted directories.


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

## Visualization ![](./visualizations/complete_quantitative_analysis.png)

## Dataset

Datasets (M3FD, MSRS, RoadScene) are used to train. You can get it from [here](https://pan.baidu.com/s/1lmLmkbbSMr_PEwZdr3HPhg?pwd=ktvq).
