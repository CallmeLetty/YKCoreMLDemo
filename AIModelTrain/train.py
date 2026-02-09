
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from model_def import ChineseVocab, ChineseClassifier
import jieba
import pickle
import pandas as pd  # 导入 Pandas

# ---------------------------------------------------------
# 2. 中文分词与词表构建
# ---------------------------------------------------------
def chinese_tokenizer(text):
    # 使用 jieba 进行精确模式分词
    return list(jieba.cut(text))

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

        
if __name__ == "__main__":
# -------------------------------------------------------
    # 步骤 1: 从 CSV 加载数据
    # -------------------------------------------------------
    csv_path = 'data.csv' # 你的文件名
    print(f"正在从 {csv_path} 读取数据...")
    
    # read_csv 会自动处理表头
    # 如果你的 CSV 编码有问题，可以尝试加 encoding='utf-8' 或 'gbk'
    df = pd.read_csv(csv_path)
    
    # 检查 CSV 是否有空行，并清理掉
    df = df.dropna(subset=['label', 'review'])
    
    # 把 DataFrame 转换为之前的 list of tuples 格式: [(1, "很好"), (0, "差"), ...]
    raw_train_data = list(zip(df['label'], df['review']))
    
    print(f"成功加载 {len(raw_train_data)} 条数据。")
    # 2. 构建并保存词表 (非常重要！)
    vocab = ChineseVocab(raw_train_data)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("词表已保存至 vocab.pkl")

    # 3. 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChineseClassifier(len(vocab)).to(device)
    # ... [此处运行训练循环代码] ...
    
    # 4. 保存模型权重
    torch.save(model.state_dict(), 'chinese_model.pth')
    print("模型权重已保存至 chinese_model.pth")
