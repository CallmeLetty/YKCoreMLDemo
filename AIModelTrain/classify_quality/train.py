"""
情感 + 质量统一四分类：一个模型同时输出「情感」与「是否说得有逻辑」。

类别：positive_high（正面且优质）、positive_low（正面但偏情绪）、
      negative_high（负面但有论据）、negative_low（负面且情绪化）
使用 AIModelTrain/BaseClassify 的 model_def（ChineseVocab、ChineseClassifier），保证与主项目一致。
请在 AIModelTrain/classify_quality 目录下运行。
"""
import sys
import os
_aimodel_train = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_base_classify = os.path.join(_aimodel_train, 'BaseClassify')
if _base_classify not in sys.path:
    sys.path.insert(0, _base_classify)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model_def import ChineseVocab, ChineseClassifier
import pickle
import json
import pandas as pd

# 四类：情感(正/负) × 质量(高/低)，顺序固定，与 Core ML 导出一致
CLASS_NAMES = ['positive_high', 'positive_low', 'negative_high', 'negative_low']
LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}


class SimpleDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data[idx]
        return label, self.vocab.encode(text)


MAX_LEN = 50
PAD_ID = 1


def collate_fn(batch):
    labels, texts = [], []
    for _label, _text_ids in batch:
        labels.append(_label)
        if len(_text_ids) < MAX_LEN:
            padded = _text_ids + [PAD_ID] * (MAX_LEN - len(_text_ids))
        else:
            padded = _text_ids[:MAX_LEN]
        texts.append(padded)
    labels = torch.tensor(labels, dtype=torch.long)
    texts = torch.tensor(texts, dtype=torch.long)
    return labels, texts


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, 'data.csv')
    print(f"正在从 {csv_path} 读取数据...")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['label', 'text'])
    df['label_id'] = df['label'].map(LABEL2ID)
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)
    raw_train_data = list(zip(df['label_id'], df['text']))

    print(f"成功加载 {len(raw_train_data)} 条数据。")
    print(f"  类别及 ID: {LABEL2ID}")

    vocab = ChineseVocab(raw_train_data)
    vocab_path = os.path.join(base, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"词表已保存至 {vocab_path}")

    class_names_path = os.path.join(base, 'class_names.json')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)
    print(f"类别名已保存至 {class_names_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SimpleDataset(raw_train_data, vocab)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    num_class = len(CLASS_NAMES)
    model = ChineseClassifier(len(vocab), embed_dim=64, num_class=num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"开始在 {device} 上训练（情感+质量四分类）...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for label, text in train_loader:
            label, text = label.to(device), text.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    model_path = os.path.join(base, 'chinese_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"训练结束，模型已保存为 {model_path}。可运行 convert_to_coreml.py 转为 Core ML。")
