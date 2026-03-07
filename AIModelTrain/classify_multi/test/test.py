"""
多分类交互测试：加载词表与模型，对输入句子预测类别 (dislike / happy / like / sad)。

用法（在 MultiClassify 目录下）:
  python test/test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import jieba
import pickle
import json
from model_def import ChineseVocab, ChineseClassifier

MAX_LEN = 50
PAD_ID = 1
UNK_ID = 0


def load_resources():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base)

    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChineseClassifier(len(vocab), num_class=len(class_names)).to(device)
    model.load_state_dict(torch.load('chinese_model.pth', map_location=device))
    model.eval()

    return vocab, model, device, class_names


def predict(text, vocab, model, device, class_names):
    tokens = list(jieba.cut(text))
    ids = [vocab.stoi.get(word, UNK_ID) for word in tokens]
    if len(ids) < MAX_LEN:
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    with torch.no_grad():
        ids_tensor = torch.tensor([ids], dtype=torch.int64).to(device)
        output = model(ids_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

    return class_names[pred_class], confidence, probs[0].tolist()


if __name__ == "__main__":
    vocab, model, device, class_names = load_resources()
    print("模型和词表加载完毕！类别:", class_names)

    while True:
        user_input = input("\n请输入要测试的句子 (输入 q 退出): ")
        if user_input.strip().lower() == 'q':
            break
        if not user_input.strip():
            continue
        label, conf, probs = predict(user_input, vocab, model, device, class_names)
        print(f"预测: {label} (置信度: {conf:.4f})")
        print("  各类概率:", dict(zip(class_names, [f"{p:.4f}" for p in probs])))
