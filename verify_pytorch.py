#!/usr/bin/env python3
"""
PyTorch安装验证脚本

验证PyTorch是否正确安装并能够使用GPU
可选择是否加载实际模型进行验证
"""

import argparse
import sys

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PyTorch安装验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s                    # 基础验证（不加载模型）
  %(prog)s --load-model       # 完整验证（包括加载实际模型）
  %(prog)s --model-name BAAI/bge-reranker-base  # 指定模型名称
        """
    )
    
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="加载实际模型进行验证（需要网络连接下载模型）"
    )
    
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-reranker-base",
        help="指定要加载的模型名称（默认: BAAI/bge-reranker-base）"
    )
    
    return parser.parse_args()


def verify_pytorch():
    """验证PyTorch安装"""
    print("🔍 验证PyTorch安装...")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ CUDNN版本: {torch.backends.cudnn.version()}")
            print(f"✅ 检测到 {torch.cuda.device_count()} 个GPU设备")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
            # 简单的GPU测试
            print("\n🧪 执行简单的GPU测试...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"✅ GPU计算测试通过: {z.shape}")
            
        else:
            print("❌ CUDA不可用")
            
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
        
    try:
        import torchvision
        print(f"✅ TorchVision版本: {torchvision.__version__}")
    except ImportError:
        print("❌ TorchVision未安装")
        
    try:
        import torchaudio
        print(f"✅ TorchAudio版本: {torchaudio.__version__}")
    except ImportError:
        print("❌ TorchAudio未安装")
        
    return True


def verify_model_loading(model_name: str):
    """验证模型加载功能"""
    print(f"\n🤖 验证模型加载功能: {model_name}")
    
    try:
        print("📦 正在导入sentence-transformers...")
        from sentence_transformers import CrossEncoder
        
        print(f"⬇️  正在加载模型: {model_name}")
        print("   (首次运行可能需要下载模型，请耐心等待...)")
        
        model = CrossEncoder(model_name)
        print("✅ 模型加载成功!")
        
        # 测试模型推理
        print("\n🧪 测试模型推理...")
        test_pairs = [('如何申请信用卡？', '怎么办理信用卡？')]
        scores = model.predict(test_pairs)
        print(f"✅ 模型推理测试通过!")
        print(f"   示例分数: {scores[0]:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ sentence-transformers导入失败: {e}")
        print("💡 请确保已安装sentence-transformers: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ 模型加载或推理失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_arguments()
    
    # 基础PyTorch验证
    pytorch_success = verify_pytorch()
    
    if not pytorch_success:
        print("\n❌ PyTorch基础验证失败")
        sys.exit(1)
    
    # 如果用户选择加载模型验证
    model_success = True
    if args.load_model:
        model_success = verify_model_loading(args.model_name)
    
    # 总结
    print(f"\n{'='*50}")
    if pytorch_success and model_success:
        print("🎉 所有验证通过！")
        if args.load_model:
            print("✅ PyTorch环境和模型加载功能都正常工作")
        else:
            print("✅ PyTorch环境验证完成")
            print("💡 如需验证模型加载功能，请运行: python3 verify_pytorch.py --load-model")
    else:
        print("❌ 验证失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
