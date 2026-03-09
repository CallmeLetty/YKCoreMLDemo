"""
诊断：对 data.csv 中第一条正面、第一条负面跑模型，打印原始 logits。
与 train.py 一致：positive→0 negative→1，故 dim0=正面 dim1=负面。
期望 正面→logit[0]>logit[1]，负面→logit[1]>logit[0]。预处理使用 vocab.encode 与训练/iOS 一致。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
import pickle
from model_def import ChineseVocab, ChineseClassifier

MAX_LEN = 50
PAD_ID = 1

def run_one(text, vocab, model, device):
    # 与训练、iOS 一致：用 vocab.encode（内部会 _normalize_text 再去 jieba），否则标点会导致分词不同、ID 序列不同
    ids = vocab.encode(text)
    if len(ids) < MAX_LEN:
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.int64).to(device)
        logits = model(x)
    return logits[0].tolist()

def _label_to_id(l):
    """支持字符串或数字：negative/0/负面→0，positive/1/正面→1"""
    s = str(l).strip().lower()
    if s in ("negative", "0", "负面"): return 0
    if s in ("positive", "1", "正面"): return 1
    raise ValueError(f"未知 label: {l!r}，期望 'positive'/'negative' 或 0/1")

def main():
    df = pd.read_csv("data.csv").dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str)
    df["label_id"] = df["label"].map(lambda l: _label_to_id(l))
    device = torch.device("cpu")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = ChineseClassifier(len(vocab)).to(device)
    model.load_state_dict(torch.load("chinese_model.pth", map_location=device))
    model.eval()

    pos_row = df[df["label_id"] == 1].iloc[0]
    neg_row = df[df["label_id"] == 0].iloc[0]

    pos_logits = run_one(pos_row["text"], vocab, model, device)
    neg_logits = run_one(neg_row["text"], vocab, model, device)

    # train.py 约定：positive→0, negative→1，故 dim0=正面 dim1=负面（与 iOS ClassifierConfig ['正面','负面'] 一致）
    print("=" * 60)
    print("诊断：模型原始 logits（dim0=正面, dim1=负面，与 train.py 一致）")
    print("=" * 60)
    print(f"正面样本: {pos_row['text'][:50]}...")
    print(f"  logits: [正面(dim0), 负面(dim1)] = {pos_logits}")
    print(f"  argmax={pos_logits.index(max(pos_logits))} (期望=0 才正确)")
    print()
    print(f"负面样本: {neg_row['text'][:50]}...")
    print(f"  logits: [正面(dim0), 负面(dim1)] = {neg_logits}")
    print(f"  argmax={neg_logits.index(max(neg_logits))} (期望=1 才正确)")
    print("=" * 60)
    pos_ok = pos_logits[0] > pos_logits[1]
    neg_ok = neg_logits[1] > neg_logits[0]
    if pos_ok and neg_ok:
        print("结论: 模型学对了 (dim0=正面 dim1=负面)，ClassifierConfig 用 ['正面','负面'] 即可，与 iOS 一致。")
    elif not pos_ok and not neg_ok:
        print("结论: 两条都判成同一类，模型可能偏向某一侧。检查预处理是否与训练一致（vocab.encode）。")
    else:
        print("结论: 正/负样本中有一个判反，检查数据或预处理。")
    print("=" * 60)

if __name__ == "__main__":
    main()
