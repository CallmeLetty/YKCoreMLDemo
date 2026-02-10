//
//  SecondTabView.swift
//  YKTextClassifier
//
//  Created by Yakamoz on 2026/2/10.
//

import NaturalLanguage
import SwiftUI
import CoreML

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
        case .negative: return .orange
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
}

// 评论数据模型
struct Comment: Identifiable {
    let id = UUID()
    let content: String
    let author: String
    let date: Date
    var category: CommentCategory?

    init(content: String, author: String, date: Date = Date()) {
        self.content = content
        self.author = author
        self.date = date
    }
}

// 评论分类器
class CommentClassifier {
    private var nlModel: NLModel?
    
    init?() {
        guard let modelURL = Bundle.main.url(forResource: "YKTextClassifier", withExtension: "mlmodelc"),
              let nlModel = try? NLModel(contentsOf: modelURL) else {
            return nil
        }
        
        self.nlModel = nlModel
    }

    func classify(_ text: String) -> CommentCategory {
        guard let hypotheses = nlModel?.predictedLabelHypotheses(for: text, maximumCount: 2) else {
            return .neutral
        }

        for (label, confidence) in hypotheses {
            if confidence > 0.56 {
                return CommentCategory.init(rawValue: label) ?? .neutral
            }
        }
        return .neutral
    }
}

// 第二个Tab - 评论分类
struct SecondTabView: View {
    @State private var comments: [Comment] = []
    @State private var isLoading = false
    @State private var selectedCategory: CommentCategory? = nil
    private var classifier = CommentClassifier()

    var filteredComments: [Comment] {
        if let category = selectedCategory {
            return comments.filter { $0.category == category }
        }
        return comments
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
            VStack(spacing: 0) {
                // 统计卡片
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 15) {
                        ForEach(CommentCategory.allCases, id: \.self) { category in
                            CategoryCard(
                                category: category,
                                count: categoryCounts[category] ?? 0,
                                isSelected: selectedCategory == category
                            )
                            .onTapGesture {
                                withAnimation {
                                    selectedCategory = selectedCategory == category ? nil : category
                                }
                            }
                        }
                    }
                    .padding()
                }
                .background(Color(.systemGroupedBackground))

                // 评论列表
                if comments.isEmpty {
                    VStack(spacing: 20) {
                        Image(systemName: "text.bubble")
                            .font(.system(size: 60))
                            .foregroundColor(.gray)

                        Text("暂无评论")
                            .font(.headline)
                            .foregroundColor(.gray)

                        Button(action: loadComments) {
                            Label("加载示例评论", systemImage: "arrow.clockwise")
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        ForEach(filteredComments) { comment in
                            CommentRow(comment: comment)
                        }
                    }
                    .listStyle(PlainListStyle())
                }
            }
            .navigationTitle("评论分类")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: loadComments) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(isLoading)
                }
            }
        }
        .onAppear {
            if comments.isEmpty {
                loadComments()
            }
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
            
            // 对每条评论进行分类
            self.comments = sampleComments.map { comment in
                var classified = comment
                classified.category = self.classifier?.classify(comment.content) ?? .neutral
                return classified
            }

            isLoading = false
        }
    }
}

// 分类卡片视图
struct CategoryCard: View {
    let category: CommentCategory
    let count: Int
    let isSelected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: category.icon)
                    .font(.title2)
                    .foregroundColor(category.color)

                Spacer()

                Text("\(count)")
                    .font(.title)
                    .fontWeight(.bold)
            }

            Text(category.rawValue)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(width: 140, height: 100)
        .background(isSelected ? category.color.opacity(0.1) : Color(.systemBackground))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isSelected ? category.color : Color.clear, lineWidth: 2)
        )
        .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
    }
}

// 评论行视图
struct CommentRow: View {
    let comment: Comment

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(comment.author)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Spacer()

                if let category = comment.category {
                    HStack(spacing: 4) {
                        Image(systemName: category.icon)
                            .font(.caption)
                        Text(category.rawValue)
                            .font(.caption)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(category.color.opacity(0.2))
                    .foregroundColor(category.color)
                    .cornerRadius(8)
                }
            }

            Text(comment.content)
                .font(.body)
                .foregroundColor(.primary)
                .lineLimit(nil)

            Text(comment.date, style: .time)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 8)
    }
}

#Preview {
    SecondTabView()
}
