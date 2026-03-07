"""
将情感+质量四分类模型转为 Core ML。使用 AIModelTrain/BaseClassify 的 model_def。
请在 AIModelTrain/classify_quality 目录下运行，且已运行过 train.py。
"""
import sys
import os
_aimodel_train = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_base_classify = os.path.join(_aimodel_train, 'BaseClassify')
if _base_classify not in sys.path:
    sys.path.insert(0, _base_classify)

import json
import torch
import coremltools as ct
import pickle
from model_def import ChineseVocab, ChineseClassifier

base = os.path.dirname(os.path.abspath(__file__))
vocab_path = os.path.join(base, 'vocab.pkl')
model_path = os.path.join(base, 'chinese_model.pth')
class_names_path = os.path.join(base, 'class_names.json')

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

with open(class_names_path, 'r', encoding='utf-8') as f:
    class_names = json.load(f)
num_class = len(class_names)

model_coreml = ChineseClassifier(vocab_size, embed_dim=64, num_class=num_class)
state_dict = torch.load(model_path, map_location='cpu')
model_coreml.load_state_dict(state_dict)
model_coreml.eval()

sentence_length = 50
dummy_input = torch.randint(0, vocab_size, (1, sentence_length), dtype=torch.int64)
traced_model = torch.jit.trace(model_coreml, dummy_input)

classifier_config = ct.ClassifierConfig(class_names)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="text", shape=dummy_input.shape, dtype=int)],
    classifier_config=classifier_config,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,
)
mlpackage_path = os.path.join(base, 'ClassifyQuality.mlpackage')
mlmodel.save(mlpackage_path)
print(f"转换成功：{mlpackage_path}")

vocab_json_path = os.path.join(base, 'vocab.json')
with open(vocab_json_path, 'w', encoding='utf-8') as f:
    json.dump(vocab.stoi, f, ensure_ascii=False, indent=4)
print(f"{vocab_json_path} 已生成，可拖入 Xcode 工程。")
