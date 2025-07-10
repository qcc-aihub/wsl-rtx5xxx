#!/usr/bin/env python3
"""
GPU检测和uv sync命令生成工具

这个脚本会自动检测您的GPU型号，并为您提供相应的uv sync命令。
对于RTX 5090，将使用最新的nightly PyTorch版本。
对于其他GPU，将使用稳定版本。
"""

import argparse
import subprocess
import sys
import re
from typing import Optional, Tuple


def get_gpu_info() -> Optional[str]:
    """
    获取GPU信息
    
    Returns:
        GPU型号字符串，如果检测失败则返回None
    """
    try:
        # 尝试使用nvidia-smi获取GPU信息
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_names = result.stdout.strip().split('\n')
        return gpu_names[0] if gpu_names else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: 无法检测到NVIDIA GPU或nvidia-smi不可用")
        return None


def parse_gpu_model(gpu_name: str) -> str:
    """
    解析GPU型号并返回对应的依赖组名称
    
    Args:
        gpu_name: GPU名称字符串
        
    Returns:
        对应的依赖组名称
    """
    gpu_name_lower = gpu_name.lower()
    
    # RTX 5090系列 - 使用nightly版本
    if "rtx 5090" in gpu_name_lower or "geforce rtx 5090" in gpu_name_lower:
        return "rtx5090"
    
    # RTX 4090系列
    elif "rtx 4090" in gpu_name_lower or "geforce rtx 4090" in gpu_name_lower:
        return "rtx4090"
    
    # RTX 3090系列
    elif "rtx 3090" in gpu_name_lower or "geforce rtx 3090" in gpu_name_lower:
        return "rtx3090"
    
    # RTX 2080系列
    elif "rtx 2080" in gpu_name_lower or "geforce rtx 2080" in gpu_name_lower:
        return "rtx2080"
    
    # 其他GPU默认使用rtx4090配置（稳定版本）
    else:
        return "rtx4090"


def get_uv_sync_command(gpu_group: str) -> Tuple[str, str]:
    """
    根据GPU组获取对应的uv sync命令
    
    Args:
        gpu_group: GPU依赖组名称
        
    Returns:
        (uv_sync_command, description)
    """
    if gpu_group == "rtx5090":
        return (
            "uv sync --group rtx5090",
            "RTX 5090 - 使用最新的nightly PyTorch版本，支持最新的CUDA特性"
        )
    elif gpu_group == "rtx4090":
        return (
            "uv sync --group rtx4090", 
            "RTX 4090 - 使用稳定版本的PyTorch"
        )
    elif gpu_group == "rtx3090":
        return (
            "uv sync --group rtx3090",
            "RTX 3090 - 使用稳定版本的PyTorch"
        )
    elif gpu_group == "rtx2080":
        return (
            "uv sync --group rtx2080",
            "RTX 2080 - 使用稳定版本的PyTorch"
        )
    else:
        return (
            "uv sync",
            "默认配置 - 使用基础依赖"
        )


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="GPU检测和uv sync命令生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                    # 交互式检测和同步
  %(prog)s --auto             # 自动检测和同步（无需确认）
  %(prog)s --gpu rtx5090      # 强制使用指定GPU配置
  %(prog)s --list-gpus        # 列出支持的GPU配置
  %(prog)s --dry-run          # 只显示命令，不执行
        """
    )
    
    parser.add_argument(
        "--auto", 
        action="store_true",
        help="自动执行同步命令，无需用户确认"
    )
    
    parser.add_argument(
        "--gpu",
        choices=["rtx5090", "rtx4090", "rtx3090", "rtx2080"],
        help="强制使用指定的GPU配置，跳过自动检测"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="只显示要执行的命令，不实际执行"
    )
    
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="列出所有支持的GPU配置"
    )
    
    return parser.parse_args()


def list_supported_gpus():
    """列出支持的GPU配置"""
    print("🎯 支持的GPU配置:")
    print("  rtx5090 - RTX 5090 (最新nightly PyTorch版本)")
    print("  rtx4090 - RTX 4090 (稳定版本)")
    print("  rtx3090 - RTX 3090 (稳定版本)")
    print("  rtx2080 - RTX 2080 (稳定版本)")
    print("\n💡 使用方法:")
    print("  python3 gpu_sync.py --gpu rtx5090")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 如果用户要求列出GPU配置
    if args.list_gpus:
        list_supported_gpus()
        return
    
    print("🔍 正在检测GPU信息...")
    
    # 如果用户指定了GPU类型，跳过检测
    if args.gpu:
        gpu_group = args.gpu
        command, description = get_uv_sync_command(gpu_group)
        print(f"✅ 使用指定GPU配置: {args.gpu.upper()}")
    else:
        # 检测GPU
        gpu_name = get_gpu_info()
        
        if gpu_name:
            print(f"✅ 检测到GPU: {gpu_name}")
            gpu_group = parse_gpu_model(gpu_name)
            command, description = get_uv_sync_command(gpu_group)
        else:
            print("❌ 无法检测到GPU，将使用默认配置")
            command, description = "uv sync", "默认配置 - 使用基础依赖"
    
    print(f"\n📦 推荐的依赖配置: {description}")
    print(f"🚀 执行以下命令同步依赖:")
    print(f"   {command}")
    
    # 如果是dry-run模式，只显示命令不执行
    if args.dry_run:
        print("\n💡 这是dry-run模式，命令未实际执行")
        return
    
    # 询问是否立即执行（除非是auto模式）
    should_execute = args.auto
    if not args.auto:
        response = input(f"\n是否立即执行此命令? (y/N): ").strip().lower()
        should_execute = response in ['y', 'yes', '是']
    
    if should_execute:
        print(f"\n⚡ 正在执行: {command}")
        try:
            subprocess.run(command.split(), check=True)
            print("✅ 依赖同步完成!")
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令执行失败: {e}")
            sys.exit(1)
    else:
        print("💡 您可以手动执行上面的命令来同步依赖")


if __name__ == "__main__":
    main()