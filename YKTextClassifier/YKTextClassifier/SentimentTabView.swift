//
//  SentimentTabView.swift
//  YKTextClassifier
//
//  文本感情分类：单一 View，通过顶部 Tab 切换 Create ML / PyTorch / 原生 三种解析方式，保留 showEmoji。
//

import SwiftUI

enum TextClassifierType: String, CaseIterable {
    case createML = "CreateML"
    case pytorch = "PyTorch"
    case natural = "NaturalLanguage"

    var title: String { rawValue }
}

struct SentimentTabView: View {
    // 共用 UI 状态
    @State private var selectedSubTab: TextClassifierType = .createML
    @State private var inputText: String = ""
    @State private var isAnalyzing: Bool = false

    // 各 Tab 对应解析结果，未分析过则展示「等待分析」
    @State private var resultsByTab: [TextClassifierType: String] = [:]

    // Create ML
    @State private var predictor = YKPredictor()
    @State private var debounceTask: Task<Void, Never>?

    // 实时分析开关
    @State private var realTimeAnalyzeEnabled: Bool = false
    // showEmoji 开关（用户控制是否显示跳动表情）
    @State private var showEmojiEnabled: Bool = false
    @State private var showEmoji: Bool = false
    @State private var currentEmoji: String = ""
    @State private var emojiScale: CGFloat = 0.5

    
    /// 当前选中的 Tab 对应的结果文案
    private var displayedResult: String {
        resultsByTab[selectedSubTab] ?? "等待分析"
    }

    var body: some View {
        NavigationStack {
            ZStack {
                VStack(spacing: 20) {
                    // 顶部分段：切换解析方式
                    Picker("", selection: $selectedSubTab) {
                        ForEach(TextClassifierType.allCases, id: \.self) { tab in
                            Text(tab.title).tag(tab)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal)
                    .padding(.top, 8)

                    // 输入框（唯一）
                    TextEditor(text: $inputText)
                        .frame(height: 150)
                        .padding(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                        )
                        .padding(.horizontal)
                        .onChange(of: inputText) { _, _ in
                            if realTimeAnalyzeEnabled {
                                debounceTask?.cancel()
                                debounceTask = Task {
                                    try? await Task.sleep(nanoseconds: 500_000_000)
                                    if !Task.isCancelled {
                                        analyzeText()
                                    }
                                }
                            }
                        }

                    // 实时分析 开关
                    Toggle(isOn: $realTimeAnalyzeEnabled) {
                        Text("实时分析")
                    }
                    .padding(.horizontal)
                    // showEmoji 开关
                    Toggle(isOn: $showEmojiEnabled) {
                        Text("显示表情")
                    }
                    .padding(.horizontal)

                    // 分析按钮（唯一）
                    Button(action: { analyzeText() }) {
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

                    // 分类结果：当前 Tab 对应方式的解析结果
                    VStack(alignment: .leading, spacing: 10) {
                        Text("分类结果：")
                            .font(.headline)
                        ScrollView {
                            Text(displayedResult)
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
                .onAppear { predictor.loadCreateMLModelIfNeeded() }

                // 跳动表情层（三种方式得到正面/负面时都可触发）
                if showEmoji {
                    Text(currentEmoji)
                        .font(.system(size: 100))
                        .scaleEffect(emojiScale)
                }
            }
            .navigationTitle("文本感情分类")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    @MainActor
    private func analyzeText() {
        isAnalyzing = true
        guard let result = predictor.analyzeText(inputText, selectedSubTab) else {
            isAnalyzing = false
            return
        }
        
        storeResultAndTriggerEmoji(result.desc, score: result.confidence)
    }

    private func storeResultAndTriggerEmoji(_ result: String, score: Double) {
        resultsByTab[selectedSubTab] = result
        isAnalyzing = false
        triggerEmojiIfNeeded(result, score: score)
    }


    /// 仅当用户开启「显示表情」时，根据结果判断正面/负面并显示跳动表情
    private func triggerEmojiIfNeeded(_ result: String, score: Double) {
        guard showEmojiEnabled else { return }
        let lower = result.lowercased()
        if lower.contains("positive") || lower.contains("正面") || lower.contains("积极") || lower.contains("好评") {
            showBouncingEmoji(pos: true, score: score)
        } else if lower.contains("negative") || lower.contains("负面") || lower.contains("消极") || lower.contains("批评") {
            showBouncingEmoji(pos: false, score: score)
        }
    }

    private func showBouncingEmoji(pos: Bool, score: Double) {
        guard score > 0 else { return }
        var emoji = "😐"
        if score > 0.5 && score <= 0.75 {
            emoji = pos ? "🙂" : "🙁"
        } else if score > 0.75 {
            emoji = pos ? "😍" : "😡"
        }

        currentEmoji = emoji
        showEmoji = true
        emojiScale = 0.5
        withAnimation(.spring(response: 0.3, dampingFraction: 0.5, blendDuration: 0).repeatCount(6, autoreverses: true)) {
            emojiScale = 1.2
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            withAnimation(.easeOut(duration: 0.3)) { showEmoji = false }
        }
    }
}

#Preview {
    SentimentTabView()
}
