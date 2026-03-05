import json
import torch
import coremltools as ct
import pickle
from model_def import ChineseVocab, ChineseClassifier

# 1. 载入词表获取大小
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# 2. 实例化
# 注意：这里的 embed_dim（词向量维度） 和 num_class（分类类别数）必须和训练时定义的一致
model_coreml = ChineseClassifier(vocab_size, embed_dim=64, num_class=2)

# 3. 加载权重
# 虽然原模型是 EmbeddingBag，但权重矩阵和 Embedding 是通用的
state_dict = torch.load('chinese_model.pth', map_location='cpu')
model_coreml.load_state_dict(state_dict)
model_coreml.eval()

# 4. 准备测试输入 (Dummy Input)
# 在 iOS 端，通常固定输入长度（比如每次输入 50 个词的索引）会更稳定
sentence_length = 50
dummy_input = torch.randint(0, vocab_size, (1, sentence_length), dtype=torch.int64)

# 5. 追踪模型
traced_model = torch.jit.trace(model_coreml, dummy_input)

# 与训练约定一致：train.py 中 label 0=positive、1=negative，模型输出 dim0=正面 dim1=负面
# 使用英文 key 以便 iOS classLabel_probs["positive"]/["negative"] 能取到值
classifier_config = ct.ClassifierConfig(['positive', 'negative'])

# 6. 转换到 Core ML
# 使用 float32 计算精度，与 Python 训练时一致，避免 Core ML 默认 float16 导致输出与 Python 不同
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="text", shape=dummy_input.shape, dtype=int)],
    classifier_config=classifier_config,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
)

# 7. 保存
mlmodel.save("ChineseClassifier.mlpackage")


# 8. 将词表导出为 JSON
# vocab.stoi 是存储 {"词语": ID} 的字典
word2idx = vocab.stoi

with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(word2idx, f, ensure_ascii=False, indent=4)
    
print("vocab.json 已生成，请将其拖入 Xcode 工程。")
