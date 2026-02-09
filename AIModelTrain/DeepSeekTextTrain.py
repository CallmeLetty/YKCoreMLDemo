import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

# 1. 准备数据（这里用简单的示例数据）
# 假设我们有正面和负面情感的文本
train_texts = [
    "i love this movie it is great",
    "awesome film really enjoyed it",
    "terrible waste of time",
    "horrible acting bad plot",
    "wonderful experience",
    "disappointing ending"
]

train_labels = [1, 1, 0, 0, 1, 0]  # 1:正面, 0:负面

# 2. 构建词汇表
def build_vocab(texts, min_freq=1):
    """构建词汇表"""
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # 创建词汇表：单词 -> 索引
    vocab = {word: idx+2 for idx, (word, count) in enumerate(word_counts.items())
             if count >= min_freq}
    vocab['<pad>'] = 0  # 填充符
    vocab['<unk>'] = 1  # 未知词
    return vocab

vocab = build_vocab(train_texts)
print(f"词汇表大小: {len(vocab)}")

# 3. 文本转为向量（词袋表示）
def text_to_bow(text, vocab):
    """将文本转换为词袋向量"""
    words = text.lower().split()
    vector = [0] * len(vocab)
    for word in words:
        if word in vocab:
            vector[vocab[word]] += 1
        else:
            vector[vocab['<unk>']] += 1
    return vector

# 4. 创建数据集
train_vectors = [text_to_bow(text, vocab) for text in train_texts]
train_vectors = torch.FloatTensor(train_vectors)
train_labels = torch.FloatTensor(train_labels).view(-1, 1)

print(f"数据形状: {train_vectors.shape}")
print(f"标签形状: {train_labels.shape}")

# 5. 定义简单模型
class SimpleSentimentModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 初始化模型
model = SimpleSentimentModel(len(vocab))
print(model)

# 6. 训练模型
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_vectors)
    loss = criterion(outputs, train_labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        # 计算准确率
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == train_labels).float().mean()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}')

# 7. 测试模型
test_texts = [
    "really good film",
    "not good at all",
    "excellent performance"
]

for test_text in test_texts:
    vector = text_to_bow(test_text, vocab)
    vector_tensor = torch.FloatTensor(vector).unsqueeze(0)
    prediction = model(vector_tensor)
    sentiment = "正面" if prediction.item() > 0.5 else "负面"
    print(f"文本: '{test_text}' -> 情感: {sentiment} (置信度: {prediction.item():.3f})")

