#!/usr/bin/env python3
"""
Pipeline: 训练 PyTorch 模型 → 转为 Core ML → 同步到 iOS 工程
1. 执行 AIModelTrain/train.py
2. 执行 AIModelTrain/convert_to_coreml.py
3. 将产物同步到 YKTextClassifier/YKTextClassifier/pytorch/
"""
import os
import shutil
import subprocess
import sys

# 脚本所在目录即项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))
AIMODEL_TRAIN = os.path.join(ROOT, "AIModelTrain")
PYTORCH_TARGET = os.path.join(ROOT, "YKTextClassifier", "YKTextClassifier", "pytorch")


def run_script(name: str, script_path: str) -> None:
    """在 AIModelTrain 目录下执行 Python 脚本"""
    path = os.path.join(AIMODEL_TRAIN, script_path)
    if not os.path.isfile(path):
        print(f"错误: 未找到 {path}", file=sys.stderr)
        sys.exit(1)
    print(f">>> 执行 {name}: {script_path}")
    ret = subprocess.run(
        [sys.executable, script_path],
        cwd=AIMODEL_TRAIN,
        check=False,
    )
    if ret.returncode != 0:
        print(f"错误: {name} 退出码 {ret.returncode}", file=sys.stderr)
        sys.exit(ret.returncode)
    print(f">>> {name} 完成\n")


def sync_artifacts() -> None:
    """将 AIModelTrain 的 mlpackage 和 vocab.json 替换到 pytorch 目录"""
    # convert_to_coreml.py 当前输出为 ChineseClassifier.mlpackage
    mlpackage_src = os.path.join(AIMODEL_TRAIN, "ChineseClassifier.mlpackage")
    vocab_src = os.path.join(AIMODEL_TRAIN, "vocab.json")

    mlpackage_dst = os.path.join(PYTORCH_TARGET, "PyTextClassifier.mlpackage")
    vocab_dst = os.path.join(PYTORCH_TARGET, "vocab.json")

    if not os.path.isdir(mlpackage_src):
        print(f"错误: 未找到 {mlpackage_src}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(vocab_src):
        print(f"错误: 未找到 {vocab_src}", file=sys.stderr)
        sys.exit(1)

    # 删除目标 mlpackage 后整体拷贝（目录不能 overwrite，需先删）
    if os.path.isdir(mlpackage_dst):
        shutil.rmtree(mlpackage_dst)
    shutil.copytree(mlpackage_src, mlpackage_dst)
    print(f"已同步: PyTextClassifier.mlpackage")

    shutil.copy2(vocab_src, vocab_dst)
    print(f"已同步: vocab.json")


def main() -> None:
    print("========== Pipeline 开始 ==========\n")
    run_script("训练", "train.py")
    run_script("转 Core ML", "convert_to_coreml.py")
    print(">>> 同步产物到 iOS 工程")
    sync_artifacts()
    print("\n========== Pipeline 完成 ==========")


if __name__ == "__main__":
    main()
