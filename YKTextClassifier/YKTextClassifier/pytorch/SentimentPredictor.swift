//
//  SentimentPredictor.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//

import CoreML
import NaturalLanguage // 用于基础分词

class SentimentPredictor {
    private let model: ChineseClassifier
    private var wordToId: [String: Int] = [:]
    private let maxLength = 50 // 必须与你转换模型时设定的长度一致

    init() {
        // 1. 初始化模型
        let config = MLModelConfiguration()
        model = try! ChineseClassifier(configuration: config)

        // 2. 加载词表 JSON
        if let url = Bundle.main.url(forResource: "vocab", withExtension: "json"),
           let data = try? Data(contentsOf: url)
        {
            wordToId = try! JSONSerialization.jsonObject(with: data) as! [String: Int]
        }
    }

    func predict(text: String) -> String {
        // 1. 分词 (Swift 原生分词，或者简单的字符切分)
        // 注意：如果你的模型是用 jieba 训练的，原生的分词可能略有差异，但通常能跑通
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var tokens: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex ..< text.endIndex) { range, _ in
            tokens.append(String(text[range]))
            return true
        }

        // 2. 将词转换为 ID (查不到的用 0 或者 <unk> 的索引)
        var tokenIds = tokens.compactMap { wordToId[$0] ?? 0 }

        // 3. 补齐或截断长度到 maxLength
        if tokenIds.count < maxLength {
            tokenIds.append(contentsOf: Array(repeating: 0, count: maxLength - tokenIds.count))
        } else {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }

        // 4. 转换为模型需要的 MLMultiArray 格式
        guard let inputArray = try? MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32) else {
            return "初始化输入失败"
        }

        for (index, id) in tokenIds.enumerated() {
            inputArray[index] = id as NSNumber
        }

        // 5. 执行预测
        do {
            let output = try model.prediction(text: inputArray)
            print("-------------------------")
            print(output.classLabel)
            print(output.classLabel_probs)
            print("-------------------------")
            return "预测结果: \(output.classLabel)"
        } catch {
            return "预测出错: \(error.localizedDescription)"
        }
    }
}
