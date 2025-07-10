# GPU适配的PyTorch环境配置

[toc]

这个项目提供了基于GPU型号自动适配的PyTorch环境配置方案。

## WSL CUDA 安装指南

WSL-Ubuntu 环境下安装 CUDA，请按照以下步骤操作：

### 1. 安装 CUDA Toolkit

```bash
# 下载安装包
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/

# 更新并安装
sudo apt-get update
sudo apt-get -y install cuda
```

### 2. 配置环境变量

将以下内容添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

然后重新加载配置：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### 3. 验证安装

```bash
nvcc --version
nvidia-smi
```

更多详情请参考：[NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

## 🚀 快速开始

### 一键安装（推荐）

```bash
./setup.sh
```

### 手动操作

#### 自动检测和同步

```bash
python3 gpu_sync.py                # 交互式检测和同步
python3 gpu_sync.py --auto         # 自动执行（无需确认）
python3 gpu_sync.py --gpu rtx5090  # 强制使用指定GPU配置
python3 gpu_sync.py --dry-run      # 只显示命令，不执行
python3 gpu_sync.py --list-gpus    # 列出支持的GPU配置
```

#### 验证安装

```bash
python3 verify_pytorch.py                              # 基础验证
python3 verify_pytorch.py --load-model                 # 完整验证（包括模型加载）
python3 verify_pytorch.py --load-model --model-name BAAI/bge-reranker-base  # 指定模型
```

### 手动指定GPU类型

#### RTX 5090 (最新nightly版本)

```bash
uv sync --group rtx5090
```

#### RTX 4090 (稳定版本)

```bash
uv sync --group rtx4090
```

#### RTX 3090 (稳定版本)

```bash
uv sync --group rtx3090
```

#### RTX 2080 (稳定版本)

```bash
uv sync --group rtx2080
```

#### 基础版本 (不指定GPU)

```bash
uv sync
```

## 📦 依赖配置说明

- **RTX 5090**: 使用PyTorch nightly版本，支持最新的CUDA 12.9特性和优化
- **RTX 4090/3090/2080**: 使用稳定版本的PyTorch，确保兼容性
- **其他GPU**: 自动回退到稳定版本配置

## 🔧 手动安装PyTorch (不推荐)

如果您需要手动安装，可以参考以下命令：

### RTX 5090 (nightly版本)

```bash
pip install \
    -U \
    --pre \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu129
```

或者用uv:

```bash
uv add torch torchvision torchaudio \
    -U \
    --prerelease=allow \
    --index https://download.pytorch.org/whl/nightly/cu129 \
    --group rtx5090
```

### 其他GPU (稳定版本)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
