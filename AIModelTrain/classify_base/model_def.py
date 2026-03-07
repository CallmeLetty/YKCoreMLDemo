import re
import jieba # 中文分词库，用于将中文句子切分成词语
import torch.nn as nn # 继承 PyTorch 的 nn.Module，定义神经网络模型。

# 是否在分词前去掉标点（与推理端 VocabularyManager 保持一致时，训练效果更好）
STRIP_PUNCTUATION = True

# 中文 + 英文标点、空白（用于忽略标点符号）
_PUNCT_PATTERN = re.compile(
    r'[\s\u3000-\u303f\uff00-\uffef\u2000-\u206f'
    r'!"#\$%&\'()*+,\-./:;<=>?@\[\\\]^_`\{\}|~]+'
)


def _normalize_text(text):
    """分词前预处理：若 STRIP_PUNCTUATION 为 True 则去掉标点与多余空白。"""
    if not text:
        return text
    if STRIP_PUNCTUATION:
        return _PUNCT_PATTERN.sub('', text)
    return text.strip()


# 词表定义
class ChineseVocab:
    def __init__(self, data=None, min_freq=1):
        # <unk> (unknown): 表示未知词，索引为 0
        # <pad> (padding): 用于填充短句，索引为 1
        self.stoi = {'<unk>': 0, '<pad>': 1} # string to index：词 → 索引
        self.itos = {0: '<unk>', 1: '<pad>'} # index to string：索引 → 词

        # 如果训练数据不为空，使用 Counter 统计词频。
        if data:
            from collections import Counter
            counter = Counter()
            # 遍历数据，对每条文本分词后统计词频。data 格式为 [(label, text), ...]。
            for _, text in data:
                t = _normalize_text(text)
                if t:
                    counter.update(list(jieba.cut(t)))
            idx = 2 # 从索引 2 开始（0、1 已被特殊标记占用），为满足最小词频的词分配索引。
            for word, freq in counter.items():
                if freq >= min_freq: # 只保留出现次数 ≥ min_freq 的词
                    self.stoi[word] = idx # 添加到词表
                    self.itos[idx] = word
                    idx += 1
                
    # 将文本转为索引序列。如果词不在词表中，返回 0（即 <unk>）。
    def encode(self, text):
        t = _normalize_text(text)
        if not t:
            return []
        return [self.stoi.get(word, 0) for word in list(jieba.cut(t))]
    
    # 返回词表大小。
    def __len__(self):
        return len(self.stoi)



# 与词表、推理端一致：填充符的索引，用于在 batch 中对齐不同长度的序列
PAD_ID = 1


class ChineseClassifier(nn.Module):
    """
    中文文本分类器（基于词嵌入 + 平均池化 + 全连接）。

    结构：Embedding → 对非 PAD 位置做 mean pooling → Linear → 输出各类别 logits。
    适用于短文本情感/主题分类，与 ChineseVocab + jieba 分词配合使用。
    """

    def __init__(self, vocab_size, embed_dim=64, num_class=2):
        """
        Args:
            vocab_size: 词表大小（与 ChineseVocab 的 len() 一致）。
            embed_dim: 词向量维度，默认 64。
            num_class: 分类类别数，默认 2（如正/负情感）。
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 词 → 稠密向量
        self.fc = nn.Linear(embed_dim, num_class)             # 池化后的向量 → 各类别 logits

    def forward(self, text):
        """
        前向传播。

        Args:
            text: 已编码的输入，形状 [batch, seq_len]，值为词表索引（含 PAD_ID）。

        Returns:
            logits: 形状 [batch, num_class]，未做 softmax。
        """
        # [batch, seq_len] → [batch, seq_len, embed_dim]
        embedded = self.embedding(text)
        # 只对非 pad 位置取平均，避免大量 pad 稀释语义、导致偏向某一类
        mask = (text != PAD_ID).unsqueeze(-1).float()  # [batch, seq_len, 1]
        masked_sum = (embedded * mask).sum(dim=1)     # [batch, embed_dim]
        count = mask.sum(dim=1).clamp(min=1)          # [batch, 1]，防止除零
        pooled = masked_sum / count
        return self.fc(pooled)
