import re
import jieba
import torch.nn as nn

# 是否在分词前去掉标点（与推理端保持一致）
STRIP_PUNCTUATION = True
_PUNCT_PATTERN = re.compile(
    r'[\s\u3000-\u303f\uff00-\uffef\u2000-\u206f'
    r'!"#\$%&\'()*+,\-./:;<=>?@\[\\\]^_`\{\}|~]+'
)


def _normalize_text(text):
    if not text:
        return text
    if STRIP_PUNCTUATION:
        return _PUNCT_PATTERN.sub('', text)
    return text.strip()


# 词表定义（与 AIModelTrain 一致）
class ChineseVocab:
    def __init__(self, data=None, min_freq=1):
        self.stoi = {'<unk>': 0, '<pad>': 1}
        self.itos = {0: '<unk>', 1: '<pad>'}

        if data:
            from collections import Counter
            counter = Counter()
            for _, text in data:
                t = _normalize_text(text)
                if t:
                    counter.update(list(jieba.cut(t)))
            idx = 2
            for word, freq in counter.items():
                if freq >= min_freq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def encode(self, text):
        t = _normalize_text(text)
        if not t:
            return []
        return [self.stoi.get(word, 0) for word in list(jieba.cut(t))]

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
