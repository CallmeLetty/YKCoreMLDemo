"""
调试脚本：与 iOS 端逐项对比，排查「分词一致但模型输出不同」的问题。

用法:
  python debug_pipeline.py "这个电影真的很好看"
  python debug_pipeline.py   # 使用默认测试句

预处理与训练/iOS 一致：vocab.encode（内部 _normalize_text + jieba），再 pad/截断到 50。
输出：分词结果、token IDs、填充后的 [1,50] 序列、Python 预测结果。
在 iOS 端用同一句测试，对比 SentimentPredictor 打印的 token IDs 是否完全一致。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import jieba
import pickle
from model_def import ChineseVocab, ChineseClassifier, _normalize_text

MAX_LENGTH = 50
PAD_ID = 1
UNK_ID = 0


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "主播说得真的很好！"

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    device = torch.device("cpu")
    model = ChineseClassifier(len(vocab)).to(device)
    model.load_state_dict(torch.load("chinese_model.pth", map_location=device))
    model.eval()

    # 与训练、iOS 一致：先 _normalize_text 再 jieba，用 vocab.encode 得到 ID 序列
    normalized = _normalize_text(text)
    tokens = list(jieba.cut(normalized)) if normalized else []
    ids = vocab.encode(text)
    if len(ids) < MAX_LENGTH:
        ids_padded = ids + [PAD_ID] * (MAX_LENGTH - len(ids))
    else:
        ids_padded = ids[:MAX_LENGTH]

    print("=" * 60)
    print("【Python 端】便于与 iOS 对比（预处理=vocab.encode，与训练一致）")
    print("=" * 60)
    print(f"输入: {text}")
    print(f"归一化后: {normalized!r}")
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
    # train.py 约定：0=正面 1=负面，与 iOS ClassifierConfig ['正面','负面'] 一致
    label = "正面" if pred == 0 else "负面"
    print(f"\nPython 预测: {label} (类别 {pred}, 置信度 {conf:.4f})")
    print(f"各类概率: 正面(dim0)={probs[0][0].item():.4f}, 负面(dim1)={probs[0][1].item():.4f}")
    print("=" * 60)
    print("请在 iOS 用同一句测试，并开启 debug 打印对比上述 ID 序列是否一致。")
    print("若 ID 一致但结果仍不同，请用 convert_to_coreml.py 以 FLOAT32 重新导出模型。")
    print("=" * 60)


if __name__ == "__main__":
    main()
