////  SentimentPredictor.swift
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
    private let padId = 1  // 必须与 Python 中的 <pad> 一致
    private let unkId = 0  // 必须与词表中 <unk> 的索引一致
    
    init() {
        let config = MLModelConfiguration()
        model = try! ChineseClassifier(configuration: config)
        
        if let url = Bundle.main.url(forResource: "vocab", withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            wordToId = try! JSONSerialization.jsonObject(with: data) as! [String: Int]
        }
    }
    
    func predict(text: String) -> String {
        // 1. 分词：使用 jieba 分词（与训练时一致）
        // ⚠️ 关键修复：Python 训练时用的是 jieba.cut() 分词，不是按字符分割
        // 你需要在 iOS 中集成 jieba 分词库，或者使用 NLTokenizer
        
        // 方案 A: 使用 NLTokenizer（推荐，系统自带）
        let tokens = tokenize(text: text)
        
        // 方案 B: 如果要完全匹配 Python 的 jieba，需要集成第三方库
        // 例如：https://github.com/fxsjy/jieba-ios
        
        // 2. 转换 ID
        var tokenIds = tokens.map { wordToId[$0] ?? unkId }
        
        // 3. 截断或填充到固定长度
        if tokenIds.count < maxLength {
            tokenIds.append(contentsOf: Array(repeating: padId, count: maxLength - tokenIds.count))
        } else {
            tokenIds = Array(tokenIds.prefix(maxLength))
        }
        
        // 4. 创建输入 MLMultiArray
        guard let inputArray = try? MLMultiArray(shape: [1, maxLength as NSNumber], dataType: .int32) else {
            return "初始化输入失败"
        }
        
        // 5. 填充数据
        for (index, id) in tokenIds.enumerated() {
            inputArray[[0, index] as [NSNumber]] = NSNumber(value: id)
        }
        
        do {
            // 6. 执行预测
            let input = ChineseClassifierInput(text: inputArray)
            let output = try model.prediction(input: input)
            
            // 7. 获取结果
            let label = output.classLabel
            let confidence = output.classLabel_probs[label] ?? 0.0
            
            return "预测结果: \(label) (置信度: \(String(format: "%.2f%%", confidence * 100)))"
        } catch {
            return "预测出错: \(error.localizedDescription)"
        }
    }
    
    // 使用 NLTokenizer 进行中文分词（接近 jieba 的效果）
    private func tokenize(text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        
        var tokens: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
            let token = String(text[tokenRange])
            tokens.append(token)
            return true
        }
        
        return tokens
    }
}

// MARK: - 使用示例
/*
let predictor = SentimentPredictor()

// 测试正面评论
let result1 = predictor.predict(text: "食物很美味，服务也很好！")
print(result1)  // 预测结果: 正面 (置信度: 95.23%)

// 测试负面评论
let result2 = predictor.predict(text: "等了1小时还没上菜，太差了")
print(result2)  // 预测结果: 负面 (置信度: 87.45%)
*/
