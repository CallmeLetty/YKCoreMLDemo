import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import jieba  # 导入中文分词工具
import time

# ---------------------------------------------------------
# 1. 准备数据（模拟一些中文语料）
# ---------------------------------------------------------
# 实际应用中，你会从 CSV 或 TXT 文件中读取这些数据
raw_train_data = [
    (1, "这个播客内容很棒，学到了很多"),
    (1, "非常喜欢这个博主，声音好听"),
    (1, "质量很好，物超所值，推荐购买"),
    (1, "挺不错的，下次还会再来"),
    (1, "老师讲得非常清晰，受益匪浅"),
    (0, "这东西质量太差了，千万别买"),
    (0, "播客内容很无聊，听不下去"),
    (0, "服务态度极其恶劣，再也不来了"),
    (0, "完全是浪费时间，差评"),
    (0, "视频画质很糊，看得眼睛疼"),
] * 50 # 把数据复制50份，模拟一个小规模训练集

# ---------------------------------------------------------
# 2. 中文分词与词表构建
# ---------------------------------------------------------
def chinese_tokenizer(text):
    # 使用 jieba 进行精确模式分词
    return list(jieba.cut(text))

class ChineseVocab:
    def __init__(self, data, min_freq=1):
        counter = Counter()
        for _, text in data:
            counter.update(chinese_tokenizer(text))
        
        # <unk>代表未知词，<pad>用于填充长度
        self.stoi = {'<unk>': 0, '<pad>': 1}
        self.itos = {0: '<unk>', 1: '<pad>'}
        
        idx = 2
        for word, freq in counter.items():
            if freq >= min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
    def encode(self, text):
        return [self.stoi.get(word, 0) for word in chinese_tokenizer(text)]
    
    def __len__(self):
        return len(self.stoi)

# 初始化词表
vocab = ChineseVocab(raw_train_data)
print(f"词表构建完成，共有 {len(vocab)} 个词。")
print(f"示例分词: {chinese_tokenizer('这段播客说的不错')}")

# ---------------------------------------------------------
# 3. 数据加载器 (Dataset & DataLoader)
# ---------------------------------------------------------
class SimpleDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data[idx]
        return label, self.vocab.encode(text)

def collate_fn(batch):
    labels, texts, offsets = [], [], [0]
    for _label, _text_ids in batch:
        labels.append(_label)
        t = torch.tensor(_text_ids, dtype=torch.int64)
        texts.append(t)
        offsets.append(t.size(0))
    
    labels = torch.tensor(labels, dtype=torch.float32)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    texts = torch.cat(texts)
    return labels, texts, offsets

train_ds = SimpleDataset(raw_train_data, vocab)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)

# ---------------------------------------------------------
# 4. 搭建模型 (和之前一样)
# ---------------------------------------------------------
class ChineseClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChineseClassifier(len(vocab)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 换个更好的优化器 Adam

# ---------------------------------------------------------
# 5. 训练
# ---------------------------------------------------------
print("开始训练中文模型...")
model.train()
for epoch in range(20): # 练20轮
    total_loss = 0
    for label, text, offsets in train_loader:
        label, text, offsets = label.to(device), text.to(device), offsets.to(device)
        
        optimizer.zero_grad()
        output = model(text, offsets).squeeze()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# ---------------------------------------------------------
# 6. 测试
# ---------------------------------------------------------
def predict_chinese(text):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(vocab.encode(text), dtype=torch.int64).to(device)
        offset = torch.tensor([0]).to(device)
        output = model(ids, offset)
        prob = torch.sigmoid(output).item()
        return "正面 😄" if prob > 0.5 else "负面 😡", prob

print("\n--- 测试开始 ---")
texts = ["这段播客说的不错", "这个东西真心不好用", "博主讲得很到位"]
for t in texts:
    res, prob = predict_chinese(t)
    print(f"句子: {t} -> 预测: {res} (概率: {prob:.4f})")
