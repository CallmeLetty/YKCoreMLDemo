//
//  YKPredictor.swift
//  YKTextClassifier
//
//  Created by Yakamoz on 2026/3/1.
//

import CoreML
import NaturalLanguage

enum YKClassifierType: String {
    case positive, negative, neutral, unknown
}

struct YKClassifierResult {
    var resultType: YKClassifierType
    var confidence: Double
    var desc: String
    
    init(resultType: YKClassifierType, confidence: Double, desc: String) {
        self.resultType = resultType
        self.confidence = confidence
        self.desc = desc
    }
}

class YKPredictor {
    
    private var nlModel: NLModel?
    private let predictor = PyPredictor()
    
    @MainActor
    func analyzeText(_ inputText: String, _ analyzeType: TextClassifierType) -> YKClassifierResult? {
        guard !inputText.isEmpty else { return nil }

        switch analyzeType {
        case .createML:
            return analyzeWithCreateML(inputText)
        case .pytorch:
            return analyzeWithPyTorch(inputText)
        case .natural:
            return analyzeWithNatural(inputText)
        }
    }
    
    func loadCreateMLModelIfNeeded() {
        guard nlModel == nil,
              let modelURL = Bundle.main.url(forResource: "MLTextClassifier", withExtension: "mlmodelc") else { return }
        nlModel = try? NLModel(contentsOf: modelURL)
    }
    
    // Create ML：NLModel
    private func analyzeWithCreateML(_ inputText: String) -> YKClassifierResult {
        guard let nlModel else {
            return .init(resultType: .unknown, confidence: 0, desc: "CreateML 模型未加载")
        }

        let hypotheses = nlModel.predictedLabelHypotheses(for: inputText, maximumCount: 2)
        var result: YKClassifierResult?
        for (label, confidence) in hypotheses {
            let text = "标签: \(label)\n置信度: \(String(format: "%.2f", confidence))"
            if confidence > 0.56,
                let resultType = YKClassifierType(rawValue: label) {
                result = .init(resultType: resultType,
                               confidence: confidence,
                               desc: text)
            }
        }
        guard let result else {
            return .init(resultType: .neutral, confidence: 0.5, desc: "中立")
        }
        return result
    }
    
    // PyTorch：PyPredictor
    private func analyzeWithPyTorch(_ inputText: String) -> YKClassifierResult {
        do {
            let (resultType, score) = try predictor.predict(text: inputText)
            let desc = "标签: \(resultType.rawValue)\n置信度: \(String(format: "%.2f", score))"
            return .init(resultType: resultType, confidence: score, desc: desc)
        } catch {
            return .init(resultType: .unknown, confidence: 0, desc: (error as? PyError)?.desc ?? "解析失败")
        }
    }

    // 原生：NLTagger sentimentScore（英文可靠；中文建议用 Create ML / PyTorch）
    private func analyzeWithNatural(_ inputText: String) -> YKClassifierResult {
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = inputText

        // 先检测语言，再设置 tagger：若强制用 .simplifiedChinese，英文如 "very good" 会被误判为负
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(inputText)
        if let dominant = recognizer.dominantLanguage {
            tagger.setLanguage(dominant, range: inputText.startIndex..<inputText.endIndex)
        }
        // 若不支持当前语言的 sentiment，availableTagSchemes 可能不包含 .sentimentScore，但 tag 仍可能返回

        let (tag, _) = tagger.tag(at: inputText.startIndex,
                                  unit: .paragraph,
                                  scheme: .sentimentScore)

        let desc: String
        let score: Double
        let type: YKClassifierType
        if let tag = tag,
            let confidence = Double(tag.rawValue) {
            let label = naturalSentimentLabel(score: confidence)
            desc = "标签:\(label)\n置信度:\(confidence)"
            score = confidence

            if confidence == 0 {
                type = .neutral
            } else if confidence > 0 {
                type = .positive
            } else {
                type = .negative
            }
        } else {
            desc = "解析失败（可尝试用英文或改用 Create ML / PyTorch）"
            score = 0
            type = .unknown
        }
        return .init(resultType: type, confidence: score, desc: desc)
    }

    private func naturalSentimentLabel(score: Double) -> String {
        if score > 0.5 { return "非常积极 😍" }
        if score > 0.1 { return "积极 🙂" }
        if score < -0.5 { return "非常消极 😡" }
        if score < -0.1 { return "消极 🙁" }
        return "中性 😐"
    }
}
