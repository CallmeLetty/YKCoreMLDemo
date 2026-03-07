import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import jieba
import pickle
from model_def import ChineseVocab, ChineseClassifier

# 加载词表和模型
def load_resources():
    # 1. 加载词表
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # 2. 初始化模型并加载权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChineseClassifier(len(vocab)).to(device)
    model.load_state_dict(torch.load('chinese_model.pth', map_location=device))
    model.eval() # 开启预测模式
    
    return vocab, model, device

def predict(text, vocab, model, device):
    with torch.no_grad():
        # 使用加载的词表进行编码
        tokens = list(jieba.cut(text))
        ids = [vocab.stoi.get(word, 0) for word in tokens]
        
        # 转换为张量，添加 batch 维度 [1, seq_len]
        ids_tensor = torch.tensor([ids], dtype=torch.int64).to(device)
        
        # 模型输出 [1, num_class]
        output = model(ids_tensor)
        
        # 使用 softmax 获取概率分布
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        return "正面 😄" if pred_class == 1 else "负面 😡", confidence

if __name__ == "__main__":
    vocab, model, device = load_resources()
    print("模型和词表加载完毕！")
    
    while True:
        user_input = input("\n请输入要测试的句子 (输入 q 退出): ")
        if user_input.lower() == 'q':
            break
        
        res, prob = predict(user_input, vocab, model, device)
        print(f"预测结果: {res} (可靠度: {prob:.4f})")

