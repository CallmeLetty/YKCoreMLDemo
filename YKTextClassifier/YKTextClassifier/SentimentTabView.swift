//
//  SentimentTabView.swift
//  YKTextClassifier
//
//  文本感情分类：单一 View，通过顶部 Tab 切换 Create ML / PyTorch / 原生 三种解析方式，保留 showEmoji。
//

import SwiftUI
import UIKit

enum TextClassifierType: String, CaseIterable {
    case createML = "CreateML"
    case pytorch = "PyTorch"
    case natural = "NaturalLanguage"

    var title: String { rawValue }
}

struct SentimentTabView: View {
    @Environment(\.isDarkMode) private var isDarkMode
    // 共用 UI 状态
    @State private var selectedSubTab: TextClassifierType = .createML
    @State private var inputText: String = ""
    @State private var isAnalyzing: Bool = false

    // 各 Tab 对应解析结果，未分析过则展示「等待分析」
    @State private var resultsByTab: [TextClassifierType: YKClassifierResult] = [:]

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
    @State private var resultVisible: Bool = false
    @State private var analyzeButtonPressed: Bool = false

    
    /// 当前选中的 Tab 对应的结果文案
    private var displayedResult: String {
        resultsByTab[selectedSubTab]?.desc ?? "等待分析"
    }

    var body: some View {
        NavigationStack {
            ZStack {
                // 背景渐变（随设置页日/夜间切换）
                AppTheme.backgroundGradient(dark: isDarkMode)
                    .ignoresSafeArea()

                VStack(spacing: 24) {
                    // 顶部分段：切换解析方式
                    Picker("", selection: $selectedSubTab) {
                        ForEach(TextClassifierType.allCases, id: \.self) { tab in
                            Text(tab.title).tag(tab)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal, 20)
                    .padding(.top, 12)
                    .onAppear {
                        let normalColor = isDarkMode ? UIColor.white.withAlphaComponent(0.85) : UIColor.darkGray
                        let selectedColor = UIColor.purple
                        UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: normalColor], for: .normal)
                        UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: selectedColor], for: .selected)
                    }
                    .onChange(of: isDarkMode) { _, _ in
                        let normalColor = isDarkMode ? UIColor.white.withAlphaComponent(0.85) : UIColor.darkGray
                        let selectedColor = UIColor.purple
                        UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: normalColor], for: .normal)
                        UISegmentedControl.appearance().setTitleTextAttributes([.foregroundColor: selectedColor], for: .selected)
                    }

                    // 输入框（卡片化 + 动效）
                    VStack(alignment: .leading, spacing: 8) {
                        Text("输入文本")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                        TextEditor(text: $inputText)
                            .frame(height: 140)
                            .scrollContentBackground(.hidden)
                            .foregroundColor(isDarkMode ? .white : .primary)
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
                    }
                    .padding(16)
                    .cardStyle()
                    .padding(.horizontal, 20)

                    // 开关行
                    HStack(spacing: 20) {
                        Toggle(isOn: $realTimeAnalyzeEnabled) {
                            Text("实时分析")
                                .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                        }
                        .tint(Color(red: 0.55, green: 0.45, blue: 1.0))
                        Toggle(isOn: $showEmojiEnabled) {
                            Text("显示表情")
                                .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                        }
                        .tint(Color(red: 0.55, green: 0.45, blue: 1.0))
                    }
                    .padding(.horizontal, 20)

                    // 分析按钮（渐变 + 按压动效）
                    Button(action: {
                        withAnimation(AppAnimation.springQuick) { analyzeButtonPressed = true }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                            analyzeText()
                            withAnimation(AppAnimation.springQuick) { analyzeButtonPressed = false }
                        }
                    }) {
                        HStack(spacing: 10) {
                            if isAnalyzing {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                    .scaleEffect(0.9)
                            }
                            Text(isAnalyzing ? "分析中..." : "分析文本")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            Group {
                                if inputText.isEmpty {
                                    RoundedRectangle(cornerRadius: 14)
                                        .fill(Color.gray.opacity(0.5))
                                } else {
                                    RoundedRectangle(cornerRadius: 14)
                                        .fill(LinearGradient(
                                            colors: [Color(red: 0.45, green: 0.35, blue: 1.0), Color(red: 0.6, green: 0.4, blue: 1.0)],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        ))
                                }
                            }
                        )
                        .foregroundColor(.white)
                        .shadow(color: (inputText.isEmpty ? .clear : Color.purple.opacity(0.4)), radius: 12, y: 4)
                    }
                    .buttonStyle(.plain)
                    .scaleEffect(analyzeButtonPressed ? 0.97 : 1.0)
                    .disabled(inputText.isEmpty || isAnalyzing)
                    .padding(.horizontal, 20)

                    // 分类结果（卡片 + 入场动效）
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("分类结果")
                                .fontWeight(.semibold)
                                .font(.headline)
                                .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                            Spacer()
                        }
                        ScrollView {
                            Text(displayedResult)
                                .frame(maxWidth: .infinity,
                                       alignment: .leading)
                                .font(.subheadline)
                                .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                        }
                        .frame(height: 140)
                        .opacity(resultVisible ? 1 : 0)
                        .offset(y: resultVisible ? 0 : 8)
                    }
                    .padding(16)
                    .cardStyle()
                    .padding(.horizontal, 20)
                    .onChange(of: displayedResult) { _, _ in
                        withAnimation(AppAnimation.springSmooth) { resultVisible = true }
                    }

                    Spacer()
                }
                .padding(.vertical, 8)
                .onAppear {
                    predictor.loadCreateMLModelIfNeeded()
                    withAnimation(AppAnimation.springSmooth.delay(0.15)) { resultVisible = true }
                }

                // 跳动表情层
                if showEmoji {
                    Text(currentEmoji)
                        .font(.system(size: 100))
                        .scaleEffect(emojiScale)
                        .shadow(color: .black.opacity(0.3), radius: 20)
                }
            }
            .navigationTitle("文本感情分类")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(isDarkMode ? .dark : .light, for: .navigationBar)
        }
    }
    
    @MainActor
    private func analyzeText() {
        isAnalyzing = true
        guard let result = predictor.analyzeText(inputText, selectedSubTab) else {
            isAnalyzing = false
            return
        }
        resultsByTab[selectedSubTab] = result
        isAnalyzing = false
        triggerEmojiIfNeeded(result.resultType, score: result.confidence)
    }

    /// 仅当用户开启「显示表情」时，根据结果判断正面/负面并显示跳动表情
    private func triggerEmojiIfNeeded(_ type: YKClassifierType, score: Double) {
        guard showEmojiEnabled,
        [YKClassifierType.positive, .negative].contains(type) else { return }
        showBouncingEmoji(pos: type == .positive, score: score)
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
