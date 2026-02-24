//
//  SentimentPredictor.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//
import CoreML
//import NaturalLanguage
import YKJiebaSupport

class SentimentPredictor {
    /// 设为 true 时在控制台打印 token IDs，便于与 Python debug_pipeline.py 输出对比
    private let debugTokenIds = false

    private let model: ChineseClassifier
    private var wordToId: [String: Int] = [:]
    private let maxLength = 50
    private let padId = 1  // 必须与 Python 中的 [1] * ... 一致
    private let unkId = 0  // 必须与词表中 <unk> 的索引一致

    init() {
        let config = MLModelConfiguration()
        // 强制使用 CPU 推理，与 Python float32 数值一致（避免 Neural Engine/GPU 的精度差异）
        config.computeUnits = .cpuOnly
        // 注意：这里的 ChineseClassifier 是由 CoreML 工具根据你的 .mlmodel 自动生成的类
        model = try! ChineseClassifier(configuration: config)

        if let url = Bundle.main.url(forResource: "vocab", withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            wordToId = try! JSONSerialization.jsonObject(with: data) as! [String: Int]
        }
    }

    func predict(text: String) -> String {
        // 1. 分词：建议与训练时保持一致（如果训练是用字，这里也用字）
//        let tokens = text.map { String($0) }
        let tokens = tokenizer(text)
        
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

        if debugTokenIds {
            print("[SentimentPredictor] 分词: \(tokens)")
            print("[SentimentPredictor] 填充后 tokenIds (共 \(tokenIds.count)): \(tokenIds)")
        }

        do {
            // 这里 text 必须和转换脚本里的 ct.TensorType(name="text", ...) 名字一致
            let input = ChineseClassifierInput(text: inputArray)
            let output = try model.prediction(input: input)
            
            // Core ML 的 ClassifierConfig 暴露的 classLabel_probs 实际是 logits（未做 softmax），
            // 与 Python 端 torch.softmax(output, dim=1) 不一致，需在 iOS 端做 softmax 得到 0~1 概率
            let negLogit = output.classLabel_probs["负面"] ?? 0.0
            let posLogit = output.classLabel_probs["正面"] ?? 0.0
            let (negProb, posProb) = softmax(neg: negLogit, pos: posLogit)
            
            let label = output.classLabel
            let confidence = label == "负面" ? negProb : posProb
            return "预测结果: \(label)，置信度 \(String(format: "%.4f", confidence))"
        } catch {
            return "预测出错: \(error.localizedDescription)"
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

    func tokenizer(_ text: String) -> [String] {
        return JiebaBridge.shared().cut(text, useHMM: true)
    }
//    func tokenizer(_ text: String) -> [String] {
//        let tokenizer = NLTokenizer(unit: .word)
//        tokenizer.string = text
//        
//        var words: [String] = []
//        
//        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
//            words.append(String(text[tokenRange]))
//            return true
//        }
//        
//        return words
//    }
    
    

}
