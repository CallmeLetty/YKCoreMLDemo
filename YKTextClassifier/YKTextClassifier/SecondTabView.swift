//
//  SecondTabView.swift
//  YKTextClassifier
//
//  Created by Yakamoz on 2026/2/10.
//

import SwiftUI
import NaturalLanguage

// 评论分类类型
enum CommentCategory: String, CaseIterable {
    case positive = "好评"
    case negative = "批评建议"
    case neutral = "中立讨论"
    
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
    static func classify(_ text: String) -> CommentCategory {
        // 使用 NaturalLanguage 框架进行情感分析
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text
        
        let (sentiment, _) = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore)
        
        if let sentimentValue = sentiment {
            let score = Double(sentimentValue.rawValue) ?? 0.0
            
            // 根据情感分数分类
            if score > 0.3 {
                return .positive
            } else if score < -0.1 {
                return .negative
            } else {
                return .neutral
            }
        }
        
        // 关键词辅助判断
        let positiveKeywords = ["好", "棒", "优秀", "喜欢", "满意", "推荐", "赞", "不错", "完美", "优质"]
        let negativeKeywords = ["差", "烂", "问题", "bug", "建议", "改进", "失望", "糟糕", "不好", "缺点"]
        
        let lowercased = text.lowercased()
        let positiveCount = positiveKeywords.filter { lowercased.contains($0) }.count
        let negativeCount = negativeKeywords.filter { lowercased.contains($0) }.count
        
        if positiveCount > negativeCount {
            return .positive
        } else if negativeCount > positiveCount {
            return .negative
        } else {
            return .neutral
        }
    }
}

// 第二个Tab - 评论分类
struct SecondTabView: View {
    @State private var comments: [Comment] = []
    @State private var isLoading = false
    @State private var selectedCategory: CommentCategory? = nil
    
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
                Comment(content: "这个应用真的太好用了！界面设计很漂亮，功能也很强大，强烈推荐！", author: "张三"),
                Comment(content: "用了一段时间，整体不错，但是希望能增加夜间模式功能。", author: "李四"),
                Comment(content: "分类准确率很高，对我的工作帮助很大，五星好评！", author: "王五"),
                Comment(content: "应用经常闪退，希望尽快修复这个bug，影响使用体验。", author: "赵六"),
                Comment(content: "功能还可以，但是加载速度有点慢，建议优化一下性能。", author: "钱七"),
                Comment(content: "非常满意，开发团队很用心，期待更多功能更新。", author: "孙八"),
                Comment(content: "界面设计一般，功能也比较基础，没有什么特别之处。", author: "周九"),
                Comment(content: "完美的应用！解决了我的痛点，值得购买会员。", author: "吴十"),
                Comment(content: "有一些小问题，比如文字分类有时不太准确，希望改进算法。", author: "郑十一"),
                Comment(content: "中规中矩的应用，能满足基本需求，价格合理。", author: "冯十二"),
                Comment(content: "太棒了！这是我用过最好的文本分类工具，效率提升明显。", author: "陈十三"),
                Comment(content: "建议增加批量处理功能，现在一条条处理太慢了。", author: "褚十四"),
                Comment(content: "还行吧，没有特别惊艳，也没有明显缺点。", author: "卫十五"),
                Comment(content: "客服响应很快，问题解决及时，体验很好！", author: "蒋十六"),
                Comment(content: "价格有点贵，功能和同类产品差不多，性价比不高。", author: "沈十七")
            ]
            
            // 对每条评论进行分类
            self.comments = sampleComments.map { comment in
                var classified = comment
                classified.category = CommentClassifier.classify(comment.content)
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
            
            Text(comment.date, style: .relative)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 8)
    }
}


#Preview {
    SecondTabView()
}
