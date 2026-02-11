//
//  SentimentPredictor.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//
import CoreML
import NaturalLanguage

class SentimentPredictor {
    private let model: ChineseClassifier
    private var wordToId: [String: Int] = [:]
    private let maxLength = 50
    private let padId = 1  // 必须与 Python 中的 [1] * ... 一致
    private let unkId = 0  // 必须与词表中 <unk> 的索引一致

    init() {
        let config = MLModelConfiguration()
        // 注意：这里的 ChineseClassifier 是由 CoreML 工具根据你的 .mlmodel 自动生成的类
        model = try! ChineseClassifier(configuration: config)

        if let url = Bundle.main.url(forResource: "vocab", withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            wordToId = try! JSONSerialization.jsonObject(with: data) as! [String: Int]
        }
    }

    func predict(text: String) -> String {
        // 1. 分词：建议与训练时保持一致（如果训练是用字，这里也用字）
        let tokens = text.map { String($0) }

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
            return "初始化输入失败"
        }

        // 4. 填充数据
        for (index, id) in tokenIds.enumerated() {
            // 2D 数组的正确赋值方式 [0, index]
            inputArray[[0, index] as [NSNumber]] = id as NSNumber
        }

        do {
            // 这里 text 必须和转换脚本里的 ct.TensorType(name="text", ...) 名字一致
            let input = ChineseClassifierInput(text: inputArray)
            let output = try model.prediction(input: input)
            
            // 因为用了 ClassifierConfig，可以直接取 classLabel
            let label = output.classLabel
            let confidence = output.classLabel_probs[label] ?? 0.0
            return "预测结果: \(output.classLabel)"
        } catch {
            return "预测出错: \(error.localizedDescription)"
        }
    }
}
