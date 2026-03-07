"""
诊断：对 data.csv 中每个类别各取第一条样本跑模型，打印原始 logits。
用于确认输出维度与标签一致：期望每条样本的 argmax 等于其真实类别 ID。
"""
import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
_base = os.path.dirname(_here)  # classify_quality
_aimodel_train = os.path.dirname(_base)  # AIModelTrain
_base_classify = os.path.join(_aimodel_train, 'BaseClassify')
if _base_classify not in sys.path:
    sys.path.insert(0, _base_classify)

import torch
import pandas as pd
import pickle
import json
from model_def import ChineseVocab, ChineseClassifier

MAX_LEN = 50
PAD_ID = 1
UNK_ID = 0


def run_one(text, vocab, model, device):
    ids = vocab.encode(text)
    if len(ids) < MAX_LEN:
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    with torch.no_grad():
        x = torch.tensor([ids], dtype=torch.int64).to(device)
        logits = model(x)
    return logits[0].tolist()


def main():
    base = _base
    os.chdir(base)

    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    label2id = {name: i for i, name in enumerate(class_names)}

    df = pd.read_csv("data.csv").dropna(subset=["label", "text"])
    device = torch.device("cpu")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    model = ChineseClassifier(len(vocab), embed_dim=64, num_class=len(class_names)).to(device)
    model.load_state_dict(torch.load("chinese_model.pth", map_location=device))
    model.eval()

    print("=" * 60)
    print("诊断：各类别首条样本的原始 logits（顺序与 class_names 一致）")
    print("=" * 60)

    all_ok = True
    for name in class_names:
        subset = df[df["label"] == name]
        if subset.empty:
            print(f"  类别 {name}: 无样本，跳过")
            continue
        row = subset.iloc[0]
        text = row["text"]
        expected_id = label2id[name]
        logits = run_one(text, vocab, model, device)
        pred_id = logits.index(max(logits))
        ok = pred_id == expected_id
        if not ok:
            all_ok = False
        status = "✓" if ok else "✗"
        print(f"{status} {name} (期望 id={expected_id}): {text[:40]}...")
        print(f"    logits: {[f'{x:.2f}' for x in logits]}, argmax={pred_id}")
        print()

    print("=" * 60)
    if all_ok:
        print("结论: 各样本 argmax 与真实标签一致，模型学对了。")
    else:
        print("结论: 存在 argmax 与真实标签不一致的样本，可检查数据或增加训练轮数。")
    print("=" * 60)


if __name__ == "__main__":
    main()
