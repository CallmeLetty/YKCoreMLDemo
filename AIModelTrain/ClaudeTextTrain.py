import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# 1. 准备数据（这里用简单的示例数据）
train_data = [
    ("这个电影太好看了", 1),  # 1表示正面
    ("非常棒的体验", 1),
    ("我很喜欢这个产品", 1),
    ("质量很好值得购买", 1),
    ("太差了完全不推荐", 0),  # 0表示负面
    ("浪费时间和金钱", 0),
    ("非常失望", 0),
    ("质量太差了", 0),
]

# 2. 构建词汇表
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.idx = 2
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)

# 3. 简单的分词函数（字符级别）
def tokenize(text):
    return list(text)  # 将文本拆分成字符

# 4. 构建词汇表
vocab = Vocabulary()
for text, _ in train_data:
    for word in tokenize(text):
        vocab.add_word(word)

print(f"词汇表大小: {len(vocab)}")

# 5. 创建数据集类
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=20):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = tokenize(text)
        
        # 转换为索引
        indices = [self.vocab.word2idx.get(token, 1) for token in tokens]
        
        # 填充或截断
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices), torch.tensor(label)

# 6. 定义简单的神经网络模型
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)  # 2个类别：正面/负面
    
    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 使用最后一个隐藏状态
        output = self.fc(hidden[-1])  # [batch_size, 2]
        return output

# 7. 创建数据加载器
dataset = TextDataset(train_data, vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 8. 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentModel(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 9. 训练模型
num_epochs = 50

print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

print("训练完成！")

# 10. 测试模型
def predict(text, model, vocab, device, max_len=20):
    model.eval()
    tokens = tokenize(text)
    indices = [vocab.word2idx.get(token, 1) for token in tokens]
    
    # 填充
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    tensor = torch.tensor([indices]).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1)
    
    return "正面" if prediction.item() == 1 else "负面"

# 11. 测试几个例子
test_texts = [
    "这个真的很棒",
    "太失望了",
    "非常好",
    "很差劲"
]

print("\n预测结果:")
for text in test_texts:
    result = predict(text, model, vocab, device)
    print(f"文本: '{text}' -> 情感: {result}")
