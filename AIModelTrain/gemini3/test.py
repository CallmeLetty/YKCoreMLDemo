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
        
        ids_tensor = torch.tensor(ids, dtype=torch.int64).to(device)
        offsets = torch.tensor([0]).to(device)
        
        output = model(ids_tensor, offsets)
        prob = torch.sigmoid(output).item()
        return "正面 😄" if prob > 0.5 else "负面 😡", prob

if __name__ == "__main__":
    vocab, model, device = load_resources()
    print("模型和词表加载完毕！")
    
    while True:
        user_input = input("\n请输入要测试的句子 (输入 q 退出): ")
        if user_input.lower() == 'q':
            break
        
        res, prob = predict(user_input, vocab, model, device)
        print(f"预测结果: {res} (可靠度: {prob:.4f})")

