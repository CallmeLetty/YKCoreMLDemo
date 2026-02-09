import jieba

# 把词表定义放这里
class ChineseVocab:
    def __init__(self, data=None, min_freq=1):
        self.stoi = {'<unk>': 0, '<pad>': 1}
        self.itos = {0: '<unk>', 1: '<pad>'}
        if data:
            from collections import Counter
            counter = Counter()
            for _, text in data:
                counter.update(list(jieba.cut(text)))
            idx = 2
            for word, freq in counter.items():
                if freq >= min_freq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                
    def encode(self, text):
        return [self.stoi.get(word, 0) for word in list(jieba.cut(text))]
    
    def __len__(self):
        return len(self.stoi)

# 把模型定义放这里
import torch.nn as nn
class ChineseClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_class=2): # 参数需与原模型一致
        super().__init__()
        # 1. 使用普通的 Embedding 替换 EmbeddingBag
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 2. 这里的 fc 层名称必须和原模型一致，以便加载权重
        self.fc = nn.Linear(embed_dim, num_class)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.mean(dim=1)

        return self.fc(embedded)
