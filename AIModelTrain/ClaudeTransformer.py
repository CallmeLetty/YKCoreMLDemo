#import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#
#from transformers import pipeline
#
## 现在应该可以下载了
#classifier = pipeline("sentiment-analysis",
#                     model="uer/roberta-base-finetuned-dianping-chinese")
#
#result = classifier("背景音乐太大声，盖过人声，听得头疼。")
#print(result)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 使用更多的训练数据
train_data = [
    ("这个电影太好看了", 1),
    ("非常棒的体验", 1),
    ("我很喜欢这个产品", 1),
    ("质量很好值得购买", 1),
    ("服务态度超级好", 1),
    ("物超所值强烈推荐", 1),
    ("味道很不错", 1),
    ("环境优雅舒适", 1),
    ("性价比很高", 1),
    ("会再来的", 1),
    ("太差了完全不推荐", 0),
    ("浪费时间和金钱", 0),
    ("非常失望", 0),
    ("质量太差了", 0),
    ("服务态度恶劣", 0),
    ("不值这个价格", 0),
    ("难吃死了", 0),
    ("环境很差", 0),
    ("性价比低", 0),
    ("不会再来了", 0),
]

# 词汇表
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

# 分词
def tokenize(text):
    return list(text)

# 构建词汇表
vocab = Vocabulary()
for text, _ in train_data:
    for char in tokenize(text):
        vocab.add_word(char)

# 数据集
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=30):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = tokenize(text)
        indices = [self.vocab.word2idx.get(token, 1) for token in tokens]
        
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices), torch.tensor(label)

# 改进的模型（加入双向LSTM和Dropout）
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 2)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # 拼接前向和后向的最后隐藏状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TextDataset(train_data, vocab)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SentimentModel(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("开始训练...")
for epoch in range(100):
    model.train()
    total_loss = 0
    
    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {total_loss/len(dataloader):.4f}')

# 预测函数
def predict(text, model, vocab, device, max_len=30):
    model.eval()
    tokens = tokenize(text)
    indices = [vocab.word2idx.get(token, 1) for token in tokens]
    
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    tensor = torch.tensor([indices]).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
        confidence = probabilities[0][prediction].item()
    
    sentiment = "正面😊" if prediction.item() == 1 else "负面😞"
    return sentiment, confidence

# 测试
test_texts = [
    "这个真的很棒",
    "太失望了",
    "非常好用推荐",
    "很差劲不要买",
    "一般般吧"
]

print("\n" + "="*50)
print("预测结果:")
print("="*50)
for text in test_texts:
    sentiment, confidence = predict(text, model, vocab, device)
    print(f"文本: '{text}'")
    print(f"情感: {sentiment} (置信度: {confidence:.2%})\n")
