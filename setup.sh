#!/bin/bash
# 一键安装和验证脚本

echo "🚀 开始自动配置PyTorch环境..."

# 检测并同步依赖
echo "📦 步骤1: 检测GPU并同步依赖"
python3 gpu_sync.py --auto

if [ $? -eq 0 ]; then
    echo "✅ 依赖同步完成"
    
    # 验证安装
    echo ""
    echo "🔍 步骤2: 验证PyTorch安装"
    python3 verify_pytorch.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 环境配置完成！"
        echo ""
        echo "💡 可选操作:"
        echo "   - 运行完整验证（包括模型加载）: python3 verify_pytorch.py --load-model"
        echo "   - 查看更多GPU配置选项: python3 gpu_sync.py --help"
    else
        echo ""
        echo "❌ PyTorch验证失败，请检查安装"
        exit 1
    fi
else
    echo "❌ 依赖同步失败"
    exit 1
fi
