//
//  SecondTabView.swift
//  YKTextClassifier
//
//  Created by Yakamoz on 2026/2/10.
//

import NaturalLanguage
import SwiftUI
import CoreML
import YKJiebaSupport

// 评论分类类型
enum CommentCategory: String, CaseIterable {
    case positive = "好评"
    case negative = "批评建议"
    case neutral = "中立讨论"
    
    init?(rawValue: String) {
        if rawValue == "positive" {
            self = .positive
        } else if rawValue == "negative" {
            self = .negative
        } else {
            self = .neutral
        }
    }

    var color: Color {
        switch self {
        case .positive: return .green
        case .negative: return .red
        case .neutral: return .blue
        }
    }

    var icon: String {
        switch self {
        case .positive: return "hand.thumbsup.fill"
        case .negative: return "exclamationmark.bubble.fill"
        case .neutral: return "message.fill"
        }
    }

    /// 将 PyTextClassifier 的预测结果映射为三分类（置信度低时归为中立）
    static func fromPyPredict(label: YKClassifierType, confidence: Double, neutralThreshold: Double = 0.55) -> CommentCategory {
        if confidence < neutralThreshold { return .neutral }
        switch label {
        case .positive: return .positive
        case .negative: return .negative
        default: return .neutral
        }
    }
}

// 评论数据模型
struct Comment: Identifiable {
    let id = UUID()
    let content: String
    let author: String
    let date: Date
    var category: CommentCategory?
    /// 评论对应的节目时间点（秒），用于情感曲线图按时间段统计
    var timestampInEpisode: TimeInterval?
    /// 该条评论的分词结果（分类时得到），用于 TF-IDF 关键词提取
    var tokens: [String]?

    init(content: String, author: String, date: Date = Date(), timestampInEpisode: TimeInterval? = nil, tokens: [String]? = nil) {
        self.content = content
        self.author = author
        self.date = date
        self.timestampInEpisode = timestampInEpisode
        self.tokens = tokens
    }
}

// 第二个Tab - 评论分类
struct CommentsListView: View {
    @Environment(\.isDarkMode) private var isDarkMode
    @State private var comments: [Comment] = []
    @State private var isLoading = false
    @State private var selectedCategory: CommentCategory? = nil
    /// 当前筛选条件下提取的关键词（随选中的评论分类变化）
    @State private var painPointKeywords: [String] = []
    @State private var cardsAppeared = false
    @State private var keywordSectionAppeared = false
    private let pyPredictor = PyPredictor()

    var filteredComments: [Comment] {
        if let category = selectedCategory {
            return comments.filter { $0.category == category }
        }
        return comments
    }

    /// 当前关键词区块的标题（随选中分类变化）
    private var keywordSectionTitle: String {
        "\(selectedCategory?.rawValue ?? "全部评论")关键词"
    }

    private var keywordSectionIcon: String {
        selectedCategory?.icon ?? "text.magnifyingglass"
    }

    private var keywordSectionColor: Color {
        selectedCategory?.color ?? .orange
    }

    /// 关键词由 Jieba 层 TF-IDF 计算（传入分类时已得到的 tokens，与 cut 同源，只分词一次）
    private func extractKeywordsForCurrentFilter() {
        let documents = filteredComments.compactMap { c -> [String]? in
            let t = (c.tokens ?? []).filter { $0.count >= 2 }
            return t.isEmpty ? nil : t
        }
        if documents.isEmpty {
            painPointKeywords = []
            return
        }
        JiebaBridge.shared().setup()
        let keywords = JiebaBridge.shared()
            .extractKeywordsTFIDF(fromTokenizedDocuments: documents, topN: 10, minWordLength: 2)
        painPointKeywords = keywords
    }

    var categoryCounts: [CommentCategory: Int] {
        var counts: [CommentCategory: Int] = [:]
        for category in CommentCategory.allCases {
            counts[category] = comments.filter { $0.category == category }.count
        }
        return counts
    }

    var body: some View {
        NavigationView {
            ZStack {
                AppTheme.backgroundGradient(dark: isDarkMode)
                    .ignoresSafeArea()

            VStack(spacing: 0) {
                // 自定义大标题（保证暗黑下为白色，不依赖系统 navigationTitle）
                Text("评论分类")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 20)
                    .padding(.top, 8)
                    .padding(.bottom, 4)
                
                // 统计卡片（带入场动效）
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 15) {
                        ForEach(Array(CommentCategory.allCases.enumerated()), id: \.element) { index, category in
                            CategoryCard(
                                category: category,
                                count: categoryCounts[category] ?? 0,
                                isSelected: selectedCategory == category
                            )
                            .opacity(cardsAppeared ? 1 : 0)
                            .offset(y: cardsAppeared ? 0 : 20)
                            .animation(AppAnimation.springSmooth.delay(Double(index) * AppAnimation.staggerDelay), value: cardsAppeared)
                            .onTapGesture {
                                withAnimation(AppAnimation.springBouncy) {
                                    selectedCategory = selectedCategory == category ? nil : category
                                    extractKeywordsForCurrentFilter()
                                    keywordSectionAppeared = false
                                    withAnimation(AppAnimation.springQuick.delay(0.1)) { keywordSectionAppeared = true }
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 16)
                }
                .background(isDarkMode ? Color.black.opacity(0.2) : Color.white.opacity(0.3))

                // 关键词：根据当前选中的评论分类提取并展示
                if !painPointKeywords.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Image(systemName: keywordSectionIcon)
                                .font(.subheadline)
                                .foregroundColor(keywordSectionColor)
                            Text(keywordSectionTitle)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                        }
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(Array(painPointKeywords.enumerated()), id: \.offset) { index, word in
                                    Text(word)
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(keywordSectionColor.opacity(0.25))
                                        .foregroundColor(keywordSectionColor)
                                        .cornerRadius(10)
                                        .opacity(keywordSectionAppeared ? 1 : 0)
                                        .scaleEffect(keywordSectionAppeared ? 1 : 0.8)
                                        .animation(AppAnimation.springBouncy.delay(Double(index) * 0.03), value: keywordSectionAppeared)
                                }
                            }
                            .padding(.horizontal, 4)
                        }
                    }
                    .padding(.vertical, 12)
                    .padding(.horizontal, 20)
                    .background(isDarkMode ? Color.black.opacity(0.15) : Color.white.opacity(0.5))
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                // 评论列表
                if comments.isEmpty {
                    VStack(spacing: 24) {
                        Image(systemName: "text.bubble.fill")
                            .font(.system(size: 56))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: isDarkMode ? [.white.opacity(0.6), .white.opacity(0.3)] : [.gray.opacity(0.5), .gray.opacity(0.3)],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                        Text("暂无评论")
                            .font(.title3)
                            .fontWeight(.semibold)
                            .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                        Button(action: loadComments) {
                            Label("加载示例评论", systemImage: "arrow.clockwise")
                                .font(.headline)
                                .padding(.horizontal, 24)
                                .padding(.vertical, 14)
                                .background(
                                    RoundedRectangle(cornerRadius: 14)
                                        .fill(LinearGradient(
                                            colors: [Color(red: 0.45, green: 0.35, blue: 1.0), Color(red: 0.6, green: 0.4, blue: 1.0)],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        ))
                                )
                                .foregroundColor(.white)
                        }
                        .buttonStyle(.plain)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        ForEach(Array(filteredComments.enumerated()), id: \.element.id) { index, comment in
                            CommentRow(comment: comment)
                                .listRowBackground(isDarkMode ? Color.white.opacity(0.06) : Color.white.opacity(0.8))
                                .listRowSeparatorTint(isDarkMode ? .white.opacity(0.1) : .black.opacity(0.08))
                                .staggeredAppear(index: index)
                        }
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(isDarkMode ? .dark : .light, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: loadComments) {
                        Image(systemName: "arrow.clockwise")
                            .fontWeight(.medium)
                    }
                    .disabled(isLoading)
                }
            }
            }
        }
        .onAppear {
            if comments.isEmpty {
                loadComments()
            }
            withAnimation(AppAnimation.springSmooth.delay(0.1)) { cardsAppeared = true }
            if !painPointKeywords.isEmpty { keywordSectionAppeared = true }
        }
        .onChange(of: painPointKeywords) { _, _ in
            keywordSectionAppeared = false
            withAnimation(AppAnimation.springQuick.delay(0.05)) { keywordSectionAppeared = true }
        }
    }
    
    private func loadComments() {
        isLoading = true

        // 模拟网络请求延迟
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            // 示例评论数据
            let sampleComments = [
                Comment(content: "主播声音太治愈了，通勤路上的必备良药。", author: "张三"),
                Comment(content: "信息量大又不枯燥，听完总想立刻分享给朋友。", author: "李四"),
                Comment(content: "这是我听过最有思考深度的中文播客之一！", author: "王五"),
                Comment(content: "更新稳定、制作精良，诚意满满！", author: "赵六"),
                Comment(content: "主题选得特别好，总能戳中我关心的话题。", author: "钱七"),
                Comment(content: "不只是娱乐，更是启发，感谢你们的声音陪伴。", author: "孙八"),
                Comment(content: "节奏把控一流，从头到尾都让人沉浸其中。", author: "周九"),
                Comment(content: "听完这期，我重新审视了自己的生活方式，太有价值了！", author: "吴"),
                Comment(content: "主播的见解独到，逻辑清晰，每次都有新视角。", author: "郑一"),
                Comment(content: "音质超好，剪辑干净，细节满分！", author: "冯二"),
                Comment(content: "真正用心做内容的播客，值得被更多人听到。", author: "陈三"),
                Comment(content: "每次更新都像收到一份精神礼物。", author: "褚四"),
                Comment(content: "内容既有温度又有深度，听完心里暖暖的。", author: "卫五"),
                Comment(content: "适合深夜静静聆听，思绪跟着一起飞翔。", author: "蒋六"),
                Comment(content: "不跟风、不浮躁，坚持做有质感的内容。", author: "蒋六"),
                Comment(content: "内容太水了，感觉就是东拼西凑的网络信息，毫无深度。", author: "蒋六"),
                Comment(content: "主播语速忽快忽慢，听起来特别累，剪辑也粗糙。", author: "蒋六"),
                Comment(content: "每期都在自说自话，完全不考虑听众的真实需求。", author: "沈七"),
                Comment(content: "广告插得太频繁，正经内容还没广告长。", author: "沈七"),
                Comment(content: "音质差到像用手机在厕所录的，根本听不下去。", author: "沈七"),
                Comment(content: "更新极不稳定，追更半年才出3期，诚意何在？", author: "沈七"),
            ]
            
            // 使用 PyTextClassifier：每条评论只执行一次「分词 + 分类」，同时保留每条的分词用于 TF-IDF
            JiebaBridge.shared().setup()
            self.comments = sampleComments.map { comment in
                var classified = comment
                let (label, confidence, tokens): (YKClassifierType, Double, [String]) = {
                    guard let result = try? self.pyPredictor.predictWithTokens(text: comment.content) else {
                        // 预测失败时用 Jieba 单独分词，保证关键词栏有数据可展示
                        let fallbackTokens = JiebaBridge.shared().cut(comment.content, useHMM: true) as? [String] ?? []
                        return (.neutral, 0, fallbackTokens)
                    }
                    var (l, c, t) = result
                    if t.isEmpty, !comment.content.isEmpty {
                        let fallbackTokens = JiebaBridge.shared().cut(comment.content, useHMM: true) as? [String] ?? []
                        t = fallbackTokens
                    }
                    return (l, c, t)
                }()
                classified.category = CommentCategory.fromPyPredict(label: label, confidence: confidence)
                classified.tokens = tokens
                return classified
            }
            self.extractKeywordsForCurrentFilter()

            isLoading = false
        }
    }
}

// 分类卡片视图
struct CategoryCard: View {
    @Environment(\.isDarkMode) private var isDarkMode
    let category: CommentCategory
    let count: Int
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: category.icon)
                    .font(.title2)
                    .foregroundColor(category.color)

                Spacer()

                Text("\(count)")
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
            }

            Text(category.rawValue)
                .font(.subheadline)
                .foregroundColor(AppTheme.textSecondary(dark: isDarkMode))
        }
        .padding(16)
        .frame(width: 140, height: 100)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(isSelected ? category.color.opacity(0.8) : AppTheme.cardStrokeColor(dark: isDarkMode), lineWidth: isSelected ? 2.5 : 1)
                )
        )
        .shadow(color: (isSelected ? category.color.opacity(0.3) : .clear), radius: 12, y: 4)
    }
}

// 评论行视图
struct CommentRow: View {
    @Environment(\.isDarkMode) private var isDarkMode
    let comment: Comment

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(comment.author)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))

                Spacer()

                if let category = comment.category {
                    HStack(spacing: 6) {
                        Image(systemName: category.icon)
                            .font(.caption)
                        Text(category.rawValue)
                            .font(.caption)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(category.color.opacity(0.25))
                    .foregroundColor(category.color)
                    .cornerRadius(10)
                }
            }

            Text(comment.content)
                .font(.body)
                .foregroundColor(AppTheme.textPrimary(dark: isDarkMode).opacity(0.9))
                .lineLimit(nil)

            Text(comment.date, style: .time)
                .font(.caption)
                .foregroundColor(AppTheme.textSecondary(dark: isDarkMode))
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 4)
    }
}

#Preview {
    CommentsListView()
}
