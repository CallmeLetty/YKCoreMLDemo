import torch                          # PyTorch 核心库
import torch.nn as nn                 # 神经网络模块
from torch.utils.data import DataLoader, Dataset  # 数据加载工具
from collections import Counter       # 计数器（这里未直接使用，在 model_def 中用到）
from model_def import ChineseVocab, ChineseClassifier  # 导入你之前定义的词表和模型
import jieba                          # 中文分词
import pickle                         # 序列化工具，用于保存词表
import pandas                         # 数据处理库，用于读取 CSV


# ---------------------------------------------------------
# 2. 中文分词与词表构建
# ---------------------------------------------------------
def chinese_tokenizer(text):
    # 使用 jieba 进行精确模式分词；与 model_def 一致，会先按 STRIP_PUNCTUATION 做预处理
    from model_def import _normalize_text
    t = _normalize_text(text) if text else ''
    return list(jieba.cut(t)) if t else []

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
    # 1. 从 CSV 加载数据
    csv_path = 'data.csv'
    print(f"正在从 {csv_path} 读取数据...")
    
    df = pandas.read_csv(csv_path) # 会自动处理表头
    df = df.dropna(subset=['label', 'text'])
    # 约定：data.csv 中 negative=负面 positive=正面。支持字符串或数字标签
    def label_to_int(l):
        s = str(l).strip().lower()
        if s in ('positive', '1'): return 0
        if s in ('negative', '0'): return 1
        raise ValueError(f"未知 label: {l!r}，期望 'positive'/'negative' 或 0/1")
    df['label'] = df['label'].astype(str)
    raw_train_data = [(label_to_int(l), t) for l, t in zip(df['label'], df['text'])]
    print(f"成功加载 {len(raw_train_data)} 条数据。")

    # 2. 构建并保存词表
    vocab = ChineseVocab(raw_train_data)
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("词表已保存至 vocab.pkl")

    # 3 准备训练环境
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

    # 4. 正式开始训练循环
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
        
        # 每隔几个 epoch 打印一下进度
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 5. 保存模型权重
    torch.save(model.state_dict(), 'chinese_model.pth')
    print("训练结束，模型已成功保存为 chinese_model.pth")
