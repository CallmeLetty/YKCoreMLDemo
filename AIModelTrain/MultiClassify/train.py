import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model_def import ChineseVocab, ChineseClassifier
import jieba
import pickle
import json
import pandas as pd

# 与 data_multi.csv 中的 label 一致，顺序固定（用于映射与 Core ML 导出）
CLASS_NAMES = ['dislike', 'happy', 'like', 'sad']
LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def chinese_tokenizer(text):
    return list(jieba.cut(text))


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
    csv_path = 'data_multi.csv'
    print(f"正在从 {csv_path} 读取数据...")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['label', 'text'])
    # 将文本标签映射为 0..num_class-1
    df['label_id'] = df['label'].map(LABEL2ID)
    df = df.dropna(subset=['label_id'])
    df['label_id'] = df['label_id'].astype(int)
    raw_train_data = list(zip(df['label_id'], df['text']))

    print(f"成功加载 {len(raw_train_data)} 条数据。")
    print(f"  类别及 ID: {LABEL2ID}")

    vocab = ChineseVocab(raw_train_data)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("词表已保存至 vocab.pkl")

    with open('class_names.json', 'w', encoding='utf-8') as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)
    print("类别名已保存至 class_names.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SimpleDataset(raw_train_data, vocab)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    num_class = len(CLASS_NAMES)
    model = ChineseClassifier(len(vocab), embed_dim=64, num_class=num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"开始在 {device} 上训练...")

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

    torch.save(model.state_dict(), 'chinese_model.pth')
    print("训练结束，模型已成功保存为 chinese_model.pth")
