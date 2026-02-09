import jieba # 中文分词库，用于将中文句子切分成词语
import torch.nn as nn # 继承 PyTorch 的 nn.Module，定义神经网络模型。

# 把词表定义放这里
class ChineseVocab:
    def __init__(self, data=None, min_freq=1):
        # <unk> (unknown): 表示未知词，索引为 0
        # <pad> (padding): 用于填充短句，索引为 1
        self.stoi = {'<unk>': 0, '<pad>': 1} # string to index：词 → 索引
        self.itos = {0: '<unk>', 1: '<pad>'} # index to string：索引 → 词

        # 如果传入了训练数据，使用 Counter 统计词频。
        if data:
            from collections import Counter
            counter = Counter()
            # 遍历数据，对每条文本分词后统计词频。假设 data 格式为 [(label, text), ...]。
            for _, text in data:
                counter.update(list(jieba.cut(text)))
            idx = 2 # 从索引 2 开始（0、1 已被特殊标记占用），为满足最小词频的词分配索引。
            for word, freq in counter.items():
                if freq >= min_freq: # 只保留出现次数 ≥ min_freq 的词
                    self.stoi[word] = idx # 添加到词表
                    self.itos[idx] = word
                    idx += 1
                
    # 将文本转为索引序列。如果词不在词表中，返回 0（即 <unk>）。
    def encode(self, text):
        return [self.stoi.get(word, 0) for word in list(jieba.cut(text))]
    
    # 返回词表大小。
    def __len__(self):
        return len(self.stoi)



class ChineseClassifier(nn.Module):
    # vocab_size: 词表大小，embed_dim: 词向量维度，默认 64，num_class: 分类类别数，默认2（二分类）
    def __init__(self, vocab_size, embed_dim=64, num_class=2): # 参数需与原模型一致
        super().__init__()

        # 嵌入层：将每个词的索引映射为 embed_dim 维的向量。
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 全连接层：将嵌入向量映射到分类输出。
        self.fc = nn.Linear(embed_dim, num_class)
        
    def forward(self, text):
        embedded = self.embedding(text)   # [batch, seq_len] → [batch, seq_len, embed_dim]
        embedded = embedded.mean(dim=1)   # 对序列维度取平均 → [batch, embed_dim]
        return self.fc(embedded)          # → [batch, num_class]
