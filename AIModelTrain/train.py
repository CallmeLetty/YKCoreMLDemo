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
    
# 固定序列长度，必须与推理端（Core ML / iOS）一致，否则训练时按 batch 内最大长度、推理时按 50，会导致结果不符
MAX_LEN = 50
PAD_ID = 1  # <pad>

# 将一个批次（batch）的样本整理成模型可以接受的张量格式。
def collate_fn(batch):
    labels, texts = [], []
    for _label, _text_ids in batch:
        labels.append(_label)
        # 截断或填充到固定长度 MAX_LEN（与推理一致）
        if len(_text_ids) < MAX_LEN:
            padded = _text_ids + [PAD_ID] * (MAX_LEN - len(_text_ids))
        else:
            padded = _text_ids[:MAX_LEN]
        texts.append(padded)

    labels = torch.tensor(labels, dtype=torch.long)
    texts = torch.tensor(texts, dtype=torch.long)
    return labels, texts  # [batch], [batch, MAX_LEN]


        
if __name__ == "__main__":
    # -------------------------------------------------------
    # 步骤 1: 从 CSV 加载数据
    # -------------------------------------------------------
    csv_path = 'data.csv'
    print(f"正在从 {csv_path} 读取数据...")
    
    # read_csv 会自动处理表头
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['label', 'review'])
    # 约定：data.csv 中 0=负面 1=正面。若 diagnose_logits.py 显示「学反了」，改 True 并重新训练，且导出用 ClassifierConfig(['正面','负面'])、eval 用 effective=1-pred
    LABEL_SWAP = False
    df['label'] = df['label'].astype(int)
    if LABEL_SWAP:
        raw_train_data = [(1 - int(l), t) for l, t in zip(df['label'], df['review'])]
    else:
        raw_train_data = list(zip(df['label'], df['review']))
    # 简单校验：第一条 1 应为正面、第一条 0 应为负面
    first_1 = next((t for l, t in raw_train_data if l == 1), None)
    first_0 = next((t for l, t in raw_train_data if l == 0), None)
    print(f"成功加载 {len(raw_train_data)} 条数据。")
    print(f"  约定: 0=负面 1=正面 | 示例 正面(1): {str(first_1)[:40]}... | 负面(0): {str(first_0)[:40]}...")
    # 2. 构建并保存词表 (非常重要！)
    vocab = ChineseVocab(raw_train_data)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("词表已保存至 vocab.pkl")

    # 3. 训练模型
    # 3.1 准备训练环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化数据集和加载器
    train_ds = SimpleDataset(raw_train_data, vocab)
    # batch_size=8 意味着模型每次看 8 句话就更新一次规律
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    # 初始化模型
    model = ChineseClassifier(len(vocab)).to(device)
    # 定义损失函数（计算预测值和真实值的差距）
    criterion = nn.CrossEntropyLoss()
    # 定义优化器（根据差距来调整模型参数，Adam 是目前最常用的）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # ---------------------------------------------------------
    # D. 正式开始训练循环 (这就是你问的那部分)
    # ---------------------------------------------------------
    epochs = 20  # 整个数据集跑 20 遍
    print(f"开始在 {device} 上训练...")

    model.train() # 告诉模型：现在是训练模式
    for epoch in range(epochs):
        total_loss = 0
        for label, text in train_loader:
            # 1. 把数据移到 GPU 或 CPU
            label, text = label.to(device), text.to(device)
            # 2. 清空之前的梯度（必须做，否则误差会累积）
            optimizer.zero_grad()
            
            # 3. 前向传播：模型给出预测结果
            output = model(text)
            
            # 4. 计算误差：预测的对不对？差了多少？
            loss = criterion(output, label)
            
            # 5. 反向传播：把误差传回给每个神经元
            loss.backward()
            
            # 6. 更新参数：根据误差微调权重
            optimizer.step()
            
            total_loss += loss.item()
        
        # 每隔几个 Epoch 打印一下进度
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # ---------------------------------------------------------
    # E. 保存模型权重 (训练完后的收尾)
    # ---------------------------------------------------------
    torch.save(model.state_dict(), 'chinese_model.pth')
    print("训练结束，模型已成功保存为 chinese_model.pth")
