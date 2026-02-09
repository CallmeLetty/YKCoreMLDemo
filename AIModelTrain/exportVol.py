import pickle
import json

# 1. 加载保存的 vocab 对象
try:
    with open('vocab.pkl', 'rb') as f:
        vocab_obj = pickle.load(f)
    print("成功加载 vocab.pkl")
except FileNotFoundError:
    print("错误：找不到 vocab.pkl 文件，请确保它在当前文件夹下。")
    exit()

# 2. 获取词典映射 (Word -> ID)
# 注意：你需要确认你的 ChineseVocab 类中保存词典的属性名是什么
# 通常是 word_to_id, stoi (string to index) 或 vocab
# 我们尝试通过 dir() 查看一下，或者直接尝试常见的名称
word_dict = {}

if hasattr(vocab_obj, 'word_to_id'):
    word_dict = vocab_obj.word_to_id
elif hasattr(vocab_obj, 'stoi'):
    word_dict = vocab_obj.stoi
elif hasattr(vocab_obj, 'vocab'):
    word_dict = vocab_obj.vocab
else:
    # 如果以上都不是，打印出来看看这个对象里有什么
    print("未能自动找到词典属性，对象包含以下内容:", dir(vocab_obj))
    # 假设你的 ChineseVocab 类里有一个叫 word_to_id 的属性
    # 如果报错，请根据 dir 打印的结果修改下面的属性名
    word_dict = vocab_obj.__dict__

# 3. 导出为 JSON
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(word_dict, f, ensure_ascii=False, indent=4)

print(f"词表已成功导出为 vocab.json，包含 {len(word_dict)} 个词。")
