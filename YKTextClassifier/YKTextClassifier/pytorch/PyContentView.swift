//
//  PyContentView.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//

import SwiftUI
import CoreML

struct PyContentView: View {
    @State private var inputText: String = ""
    @State private var classificationResult: String = ""
    @State private var isAnalyzing: Bool = false
    
    let predictor = SentimentPredictor()
    
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
                    analyzeText()
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
    }
    
    // 分析文本的函数
    func analyzeText() {
        guard !inputText.isEmpty else {
            classificationResult = ""
            return
        }
        
        isAnalyzing = true
        
        let result = predictor.predict(text: inputText)
        print(result)
        classificationResult = result
        
        DispatchQueue.main.async {
            isAnalyzing = false
        }
    }
}

#Preview {
    ContentView()
}

