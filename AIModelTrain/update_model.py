import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder

# 1. 加载你现有的模型
spec = ct.utils.load_spec("ChineseClassifier.mlpackage")
builder = NeuralNetworkBuilder(spec=spec)

# 2. 检查最后一层（通常是线性层）的名字
builder.make_updatable(["dense_layer"])

# 3. 设置训练参数
# 比如使用 SGD 优化器
builder.set_sgd_optimizer(learning_rate=0.01, batch_size=1)

# 4. 定义损失函数
# 对于分类任务，通常用 CrossEntropy
builder.set_categorical_cross_entropy_loss(name="lossLayer", input="label_probs")

# 5. 重新保存为可更新模型
ct.utils.save_spec(spec, "UpdatablePyClassifier.mlpackage")

