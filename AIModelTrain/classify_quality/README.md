# 情感 + 质量统一四分类（一个模型）

用**同一个模型**同时做「情感」和「质量」判断，一次预测得到 4 类之一：

| 类别 | 含义 |
|------|------|
| `positive_high` | 正面且说得有逻辑、有条理 |
| `positive_low`  | 正面但偏情绪、碎碎念 |
| `negative_high`| 负面但有论据、有逻辑 |
| `negative_low`  | 负面且情绪化、无结构 |

与主项目共用 `AIModelTrain/BaseClassify/model_def.py` 的 `ChineseVocab`、`ChineseClassifier`（`num_class=4`），预处理与结构一致。

## 数据格式

`data.csv` 两列：`label`, `text`。`label` 为上述四类之一。

## 训练与转 Core ML

在 **本目录 `AIModelTrain/classify_quality`** 下：

```bash
python3 train.py
python3 convert_to_coreml.py
```

得到：`vocab.pkl`、`chinese_model.pth`、`class_names.json`、`ClassifyQuality.mlpackage`、`vocab.json`。

## 测试脚本

在 **`AIModelTrain`** 目录下运行（会从 `classify_quality` 读入词表与模型）：

```bash
# 交互式预测（输入句子得四类概率，输入 q 退出）
python3 classify_quality/test/test.py

# 单句调试：打印分词、ID 序列、预测与各类概率（便于与 iOS 对比）
python3 classify_quality/test/debug_pipeline.py "这期从三个角度分析了问题"

# 诊断：用 data.csv 每类首条跑模型，检查 argmax 是否与真实标签一致
python3 classify_quality/test/diagnose_logits.py
```

## App 端用法

- 用 `ClassifyQuality.mlpackage` + `vocab.json` 做一次预测，得到 4 类概率。
- 从预测类别可同时得到：
  - **情感**：`positive_high` / `positive_low` → 正面；`negative_high` / `negative_low` → 负面。
  - **质量**：`*_high` → 优质/有逻辑；`*_low` → 一般/情绪化。
- 例如：只展示「优质评论」可筛 `positive_high` 或 `negative_high` 概率高于某阈值。
