import torch                          # PyTorch 核心库
import torch.nn as nn                 # 神经网络模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
from collections import Counter       # 计数器（这里未直接使用，在 model_def 中用到）
from model_def import ChineseVocab, ChineseClassifier  # 导入你之前定义的词表和模型
import jieba                          # 中文分词
import pickle                         # 序列化工具，用于保存词表
import pandas as pd                   # 数据处理库，用于读取 CSV


# ---------------------------------------------------------
# 2. 中文分词与词表构建
# ---------------------------------------------------------
def chinese_tokenizer(text):
    # 使用 jieba 进行精确模式分词，将文本切分成词语列表
    # 这个函数在当前代码中没有被调用，因为分词逻辑已经在 ChineseVocab.encode() 中实现了。
    return list(jieba.cut(text))

# ---------------------------------------------------------
# 3. 数据加载器 (Dataset & DataLoader)
# ---------------------------------------------------------
class SimpleDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data      # 存储原始数据 [(label, text), ...]
        self.vocab = vocab    # 词表对象
        
    def __len__(self):
        return len(self.data) # 返回数据集大小，DataLoader 需要
    
    def __getitem__(self, idx):
        label, text = self.data[idx]           # 获取第 idx 条数据
        return label, self.vocab.encode(text)  # 返回标签和编码后的索引列表
    
    def collate_fn(batch):
        labels, texts = [], []
        max_len = max(len(text_ids) for _, text_ids in batch)  # 找最长序列
        
        for _label, _text_ids in batch:
            labels.append(_label)
            # 填充到相同长度
            padded = _text_ids + [1] * (max_len - len(_text_ids))  # 1 是 <pad>
            texts.append(padded)
        
        labels = torch.tensor(labels, dtype=torch.long)
        texts = torch.tensor(texts, dtype=torch.long)
        return labels, texts  # [batch], [batch, max_len]


        
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
