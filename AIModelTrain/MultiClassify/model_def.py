import jieba
import torch.nn as nn

# 词表定义（与 AIModelTrain 一致）
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


PAD_ID = 1


class ChineseClassifier(nn.Module):
    """
    中文文本多分类器（基于词嵌入 + 平均池化 + 全连接）。
    支持 4 类：dislike / happy / like / sad。
    """

    def __init__(self, vocab_size, embed_dim=64, num_class=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        mask = (text != PAD_ID).unsqueeze(-1).float()
        masked_sum = (embedded * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        pooled = masked_sum / count
        return self.fc(pooled)
