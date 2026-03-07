"""
调试脚本：与 iOS 端逐项对比，排查「分词一致但模型输出不同」的问题。

用法（在 classify_quality 目录下）:
  python3 test/debug_pipeline.py "这期从三个角度分析了问题，逻辑清晰"
  python3 test/debug_pipeline.py   # 使用默认测试句

输出：分词结果、token IDs、填充后的 [1,50] 序列、四分类预测结果及各类概率。
"""
import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.dirname(_here)
_aimodel_train = os.path.dirname(_base)
_base_classify = os.path.join(_aimodel_train, 'BaseClassify')
if _base_classify not in sys.path:
    sys.path.insert(0, _base_classify)

import torch
import jieba
import pickle
import json
from model_def import ChineseVocab, ChineseClassifier, _normalize_text

MAX_LENGTH = 50
PAD_ID = 1
UNK_ID = 0


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "这期从三个角度分析了问题，逻辑清晰"

    base = _base
    os.chdir(base)

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)

    device = torch.device("cpu")
    model = ChineseClassifier(len(vocab), embed_dim=64, num_class=len(class_names)).to(device)
    model.load_state_dict(torch.load("chinese_model.pth", map_location=device))
    model.eval()

    tokens = list(jieba.cut(_normalize_text(text)))
    ids = vocab.encode(text)
    if len(ids) < MAX_LENGTH:
        ids_padded = ids + [PAD_ID] * (MAX_LENGTH - len(ids))
    else:
        ids_padded = ids[:MAX_LENGTH]

    print("=" * 60)
    print("【Python 端】便于与 iOS 对比")
    print("=" * 60)
    print(f"输入: {text}")
    print(f"分词: {tokens}")
    print(f"Token 数: {len(tokens)}")
    print(f"原始 ID 序列: {ids}")
    print(f"填充后长度: {len(ids_padded)}")
    print(f"填充后前 20 个 ID: {ids_padded[:20]}")
    print(f"完整 [1,50] 序列 (供逐位对比):")
    print(ids_padded)

    with torch.no_grad():
        x = torch.tensor([ids_padded], dtype=torch.int64)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred].item()

    label_name = class_names[pred]
    print(f"\nPython 预测: {label_name} (类别 id={pred}, 置信度 {conf:.4f})")
    print("各类概率:", dict(zip(class_names, [f"{probs[0][i].item():.4f}" for i in range(len(class_names))])))
    print("=" * 60)
    print("请在 iOS 用同一句测试，并开启 debug 打印对比上述 ID 序列是否一致。")
    print("若 ID 一致但结果仍不同，请用 convert_to_coreml.py 以 FLOAT32 重新导出模型。")
    print("=" * 60)


if __name__ == "__main__":
    main()
