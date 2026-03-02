//
//  SentimentPredictor.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//
import CoreML
//import NaturalLanguage
import YKJiebaSupport

struct PyError: Error {
    var desc: String
    init(desc: String) {
        self.desc = desc
    }
}

class PyPredictor {
    /// 设为 true 时在控制台打印 token IDs，便于与 Python debug_pipeline.py 输出对比
    private let debugTokenIds = true

    private lazy var tokenizer = VocabularyManager()

    private let model: PyTextClassifier
    private var wordToId: [String: Int] = [:]
    private let maxLength = 50
    private let padId = 1  // 必须与 Python 中的 [1] * ... 一致
    private let unkId = 0  // 必须与词表中 <unk> 的索引一致

    init() {
        let config = MLModelConfiguration()
        // 强制使用 CPU 推理，与 Python float32 数值一致（避免 Neural Engine/GPU 的精度差异）
        config.computeUnits = .cpuOnly
        // 注意：这里的 ChineseClassifier 是由 CoreML 工具根据你的 .mlmodel 自动生成的类
        model = try! PyTextClassifier(configuration: config)
        tokenizer.initJieba()
        if let url = Bundle.main.url(forResource: "vocab", withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            wordToId = try! JSONSerialization.jsonObject(with: data) as! [String: Int]
        }
    }

    func predict(text: String) throws -> (YKClassifierType, Double) {
        // 1. 分词(与训练时保持一致用jieba)
        let tokens = tokenizer.tokenize(text)
        
        // 2. 转换 ID
        var tokenIds = tokens.map { wordToId[$0] ?? unkId }

        // 截断或填充
        if tokenIds.count < maxLength {
            tokenIds.append(contentsOf: Array(repeating: padId, count: maxLength - tokenIds.count))
        } else {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        // 3. 创建输入。转换为 MLMultiArray.注意：CoreML 转换时如果选了 Int32，这里最好显式指定
        // 注意 shape必须严格遵守转换模型时定义的形状 是 [1, 50]
        guard let inputArray = try? MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32) else {
            throw PyError(desc: "初始化输入失败")
        }

        // 4. 填充数据
        for (index, id) in tokenIds.enumerated() {
            // 2D 数组的正确赋值方式 [0, index]
            inputArray[[0, index] as [NSNumber]] = id as NSNumber
        }

        if debugTokenIds {
            print("[SentimentPredictor] 分词: \(tokens)")
            print("[SentimentPredictor] 填充后 tokenIds (共 \(tokenIds.count)): \(tokenIds)")
        }

        do {
            // 这里 text 必须和转换脚本里的 ct.TensorType(name="text", ...) 名字一致
            let input = PyTextClassifierInput(text: inputArray)
            let output = try model.prediction(input: input)
            
            // classLabel_probs 的 key 与 convert_to_coreml.py 的 ClassifierConfig 一致：negative / positive
            let negLogit = output.classLabel_probs["negative"] ?? 0.0
            let posLogit = output.classLabel_probs["positive"] ?? 0.0
            let (negProb, posProb) = softmax(neg: negLogit, pos: posLogit)
            // 根据概率自行判定标签，与 Python argmax(softmax(logits)) 一致
            let label: YKClassifierType = posProb >= negProb ? .positive : .negative
            let confidence = label == .positive ? posProb : negProb
            return (label, confidence)
        } catch {
            throw PyError(desc: "预测出错: \(error.localizedDescription)")
        }
    }

    /// 对两类 logits 做 softmax，得到与 Python 一致的 0~1 概率（数值稳定：先减最大值再 exp）
    private func softmax(neg: Double, pos: Double) -> (Double, Double) {
        let maxL = max(neg, pos)
        let eNeg = exp(neg - maxL)
        let ePos = exp(pos - maxL)
        let sum = eNeg + ePos
        return (eNeg / sum, ePos / sum)
    }
}
