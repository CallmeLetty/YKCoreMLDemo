"""
诊断：对 data.csv 中第一条正面(1)、第一条负面(0) 跑模型，打印原始 logits。
用于确认模型输出维度与标签是否一致：期望 正面→logit[1]>logit[0]，负面→logit[0]>logit[1]。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
import jieba
import pickle
from model_def import ChineseVocab, ChineseClassifier

MAX_LEN = 50
PAD_ID = 1
UNK_ID = 0

def run_one(text, vocab, model, device):
    tokens = list(jieba.cut(text))
    ids = [vocab.stoi.get(w, UNK_ID) for w in tokens]
    if len(ids) < MAX_LEN:
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.int64).to(device)
        logits = model(x)
    return logits[0].tolist()

def main():
    df = pd.read_csv("data.csv").dropna(subset=["label", "review"])
    df["label"] = df["label"].astype(int)
    device = torch.device("cpu")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = ChineseClassifier(len(vocab)).to(device)
    model.load_state_dict(torch.load("chinese_model.pth", map_location=device))
    model.eval()

    pos_row = df[df["label"] == 1].iloc[0]
    neg_row = df[df["label"] == 0].iloc[0]

    pos_logits = run_one(pos_row["review"], vocab, model, device)
    neg_logits = run_one(neg_row["review"], vocab, model, device)

    print("=" * 60)
    print("诊断：模型原始 logits（dim0=负面, dim1=正面）")
    print("=" * 60)
    print(f"正面样本(1): {pos_row['review'][:50]}...")
    print(f"  logits: [负面(dim0), 正面(dim1)] = {pos_logits}")
    print(f"  argmax={pos_logits.index(max(pos_logits))} (期望=1 才正确)")
    print()
    print(f"负面样本(0): {neg_row['review'][:50]}...")
    print(f"  logits: [负面(dim0), 正面(dim1)] = {neg_logits}")
    print(f"  argmax={neg_logits.index(max(neg_logits))} (期望=0 才正确)")
    print("=" * 60)
    pos_ok = pos_logits[1] > pos_logits[0]
    neg_ok = neg_logits[0] > neg_logits[1]
    if pos_ok and neg_ok:
        print("结论: 模型学对了 (0=负面 1=正面)，ClassifierConfig 用 ['负面','正面'] 即可。")
    elif not pos_ok and not neg_ok:
        print("结论: 两条都判成 0（负面），模型偏向预测负面，不是标签反了。请用「固定长度 50」重新训练，勿开 LABEL_SWAP。")
    else:
        print("结论: 模型学反了（正面→0 负面→1），可设 train.py 中 LABEL_SWAP=True 后重新训练。")
    print("=" * 60)

if __name__ == "__main__":
    main()
