import json
import torch
import coremltools as ct
import pickle
from model_def import ChineseVocab, ChineseClassifier

# 1. 载入词表
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# 2. 载入类别名（与 train.py 中 CLASS_NAMES 顺序一致）
with open('class_names.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)
num_class = len(class_names)

# 3. 实例化模型（与训练时一致）
model_coreml = ChineseClassifier(vocab_size, embed_dim=64, num_class=num_class)

# 4. 加载权重
state_dict = torch.load('chinese_model.pth', map_location='cpu')
model_coreml.load_state_dict(state_dict)
model_coreml.eval()

# 5. 固定输入长度（与训练 MAX_LEN 一致）
sentence_length = 50
dummy_input = torch.randint(0, vocab_size, (1, sentence_length), dtype=torch.int64)

# 6. 追踪模型
traced_model = torch.jit.trace(model_coreml, dummy_input)

# 7. 多分类 ClassifierConfig
classifier_config = ct.ClassifierConfig(class_names)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="text", shape=dummy_input.shape, dtype=int)],
    classifier_config=classifier_config,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
)

mlmodel.save("ChineseClassifierMulti.mlpackage")
print("转换成功：ChineseClassifierMulti.mlpackage")

# 导出词表供 iOS 使用
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab.stoi, f, ensure_ascii=False, indent=4)
print("vocab.json 已生成，请将其拖入 Xcode 工程。")
