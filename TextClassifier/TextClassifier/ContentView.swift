//
//  ContentView.swift
//  TextClassifier
//
//  Created by Yakamoz on 2026/2/7.
//

import SwiftUI
import CoreML
import NaturalLanguage

struct ContentView: View {
    @State private var inputText: String = ""
    @State private var classificationResult: String = ""
    @State private var isAnalyzing: Bool = false
    @State private var classifier: YKTextClassifier?
    @State private var nlModel: NLModel?
    @State private var debounceTask: Task<Void, Never>?
    
    var body: some View {
        VStack(spacing: 20) {
            Text("中文文本分类器")
                .font(.title)
                .fontWeight(.bold)
                .padding(.top)
            
            // 输入文本框
            TextEditor(text: $inputText)
                .frame(height: 150)
                .padding(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                )
                .padding(.horizontal)
                .onChange(of: inputText) { oldValue, newValue in
                    // 取消之前的预测任务
                    debounceTask?.cancel()
                    
                    // 创建新的延迟预测任务
                    debounceTask = Task {
                        // 等待0.5秒
                        try? await Task.sleep(nanoseconds: 500_000_000)
                        
                        // 检查任务是否被取消
                        if !Task.isCancelled {
                            await analyzeText()
                        }
                    }
                }
            
            // 分析按钮
            Button(action: {
                analyzeText()
            }) {
                HStack {
                    if isAnalyzing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(0.8)
                    }
                    Text(isAnalyzing ? "分析中..." : "分析文本")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(inputText.isEmpty ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .disabled(inputText.isEmpty || isAnalyzing)
            .padding(.horizontal)
            
            // 分类结果显示区域
            VStack(alignment: .leading, spacing: 10) {
                Text("分类结果：")
                    .font(.headline)
                
                ScrollView {
                    Text(classificationResult.isEmpty ? "请输入文本进行分析" : classificationResult)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(8)
                }
                .frame(height: 150)
            }
            .padding(.horizontal)
            
            Spacer()
        }
        .padding()
        .onAppear {
            // 在视图出现时加载模型
            loadModel()
        }
    }
    
    // 加载模型
    func loadModel() {
        let config = MLModelConfiguration()
        classifier = try? YKTextClassifier(configuration: config)
        
        if let modelURL = Bundle.main.url(forResource: "YKTextClassifier", withExtension: "mlmodelc") {
           nlModel = try? NLModel(contentsOf: modelURL)
        }
    }
    
    @MainActor
    func analyzeText() {
        guard !inputText.isEmpty else {
            classificationResult = ""
            return
        }
        
        guard let classifier = classifier else {
            classificationResult = "模型加载失败"
            return
        }
        
        isAnalyzing = true
        
        do {
            // 进行预测
//            let prediction = try classifier.prediction(text: inputText)
            
            // 获取分类标签
//            let label = prediction.label
//            classificationResult = "预测类别: \(label)"
            isAnalyzing = false
            
            // 1. 加载模型
            guard let nlModel else {
                return
            }

//            // 2. 获取最可能的标签（相当于之前的 output.label）
//            if let label = nlModel.predictedLabel(for: inputText) {
//                print("预测结果: \(label)")
//            }

            // 3. 重点：获取所有标签及其置信度（即使模型 Output 里没有也能拿到）
            let hypotheses = nlModel.predictedLabelHypotheses(for: inputText, maximumCount: 2)

            var result: String?
            for (label, confidence) in hypotheses {
                let text = "标签: \(label), 置信度: \(confidence)"
                print(text)
                if confidence > 0.56 {
                    result = text
                }
            }
            
            if let result {
                classificationResult = result
            } else {
                classificationResult = result ?? "未知"
            }
        } catch {
            classificationResult = "分析出错: \(error.localizedDescription)"
            isAnalyzing = false
        }
    }
}

#Preview {
    ContentView()
}
