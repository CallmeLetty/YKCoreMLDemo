//
//  CurveTabView.swift
//  YKTextClassifier
//
//  情感曲线图：按节目时间戳统计某一集在不同时间段的情感分布，
//  帮助主播发现例如「第 15 分钟负面评论增多」等规律，优化节目内容。
//

import SwiftUI
import Charts

/// 单个时间段的统计
struct SentimentSegment: Identifiable {
    let id = UUID()
    let segmentIndex: Int
    let label: String           // 如 "10:00"
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    var positiveCount: Int
    var negativeCount: Int
    var neutralCount: Int
    
    var totalCount: Int { positiveCount + negativeCount + neutralCount }
}

struct CurveTabView: View {
    /// 本集时长（秒），默认 60 分钟
    private let episodeDuration: TimeInterval = 60 * 60
    /// 时间段长度（秒），默认 5 分钟
    private let segmentLength: TimeInterval = 5 * 60
    
    @State private var episodeComments: [Comment] = []
    @State private var segments: [SentimentSegment] = []
    @State private var isLoading = true
    @State private var segmentChoice: SegmentChoice = .fiveMin
    
    private let classifier = CommentClassifier()
    
    enum SegmentChoice: String, CaseIterable {
        case fiveMin = "10分钟"
        case tenMin = "20分钟"
        
        var seconds: TimeInterval {
            switch self {
            case .fiveMin: return 10 * 60
            case .tenMin: return 20 * 60
            }
        }
    }
    
    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    VStack(spacing: 16) {
                        ProgressView()
                        Text("正在加载本集评论并分析情感…")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if segments.isEmpty {
                    emptyState
                } else {
                    chartContent
                }
            }
            .navigationTitle("情感曲线图")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(action: loadEpisodeData) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(isLoading)
                }
            }
        }
        .onAppear {
            if episodeComments.isEmpty {
                loadEpisodeData()
            }
        }
    }
    
    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 50))
                .foregroundColor(.gray)
            Text("暂无带时间戳的评论")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("本集评论需包含时间戳才能生成情感曲线")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var chartContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // 说明
                Text("按时间段统计本集评论情感分布，便于发现例如「某时段负面评论突增」等规律。")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                HStack(alignment: .center, spacing: 10) {
                    Text("时间段")
                        .font(.subheadline)
                        .foregroundColor(.primary)
                    // 时间段选择
                    Picker("时间段", selection: $segmentChoice) {
                        ForEach(SegmentChoice.allCases, id: \.self) { choice in
                            Text(choice.rawValue).tag(choice)
                        }
                    }
                    .pickerStyle(.segmented)
                    .onChange(of: segmentChoice) { _, _ in
                        recomputeSegments(segmentLength: segmentChoice.seconds)
                    }
                }
                
                // 曲线图：三条线分别表示好评、批评、中立（用数值 X 轴使 0:00 从最左侧开始）
                Chart {
                    ForEach(segments) { seg in
                        LineMark(
                            x: .value("时间", seg.segmentIndex),
                            y: .value("数量", seg.positiveCount),
                            series: .value("情感", "好评")
                        )
                        .foregroundStyle(CommentCategory.positive.color)
                        .interpolationMethod(.catmullRom)
                        .symbol(Circle())
                        
                        LineMark(
                            x: .value("时间", seg.segmentIndex),
                            y: .value("数量", seg.negativeCount),
                            series: .value("情感", "批评建议")
                        )
                        .foregroundStyle(CommentCategory.negative.color)
                        .interpolationMethod(.catmullRom)
                        .symbol(Circle())
                        
                        LineMark(
                            x: .value("时间", seg.segmentIndex),
                            y: .value("数量", seg.neutralCount),
                            series: .value("情感", "中立讨论")
                        )
                        .foregroundStyle(CommentCategory.neutral.color)
                        .interpolationMethod(.catmullRom)
                        .symbol(Circle())
                    }
                }
                .chartXScale(domain: 0 ... max(0, segments.count - 1))
                .chartXAxis {
                    AxisMarks(values: .stride(by: 1)) { value in
                        AxisGridLine()
                        AxisValueLabel {
                            if let i = value.as(Int.self), i >= 0, i < segments.count {
                                Text(segments[i].label)
                            }
                        }
                    }
                }
                .chartYAxisLabel("评论数")
                .chartXAxisLabel("节目时间")
                .frame(height: 260)
                
                // 图例
                HStack(spacing: 20) {
                    legendItem(color: CommentCategory.positive.color, text: "好评")
                    legendItem(color: CommentCategory.negative.color, text: "批评建议")
                    legendItem(color: CommentCategory.neutral.color, text: "中立讨论")
                }
                .padding(.vertical, 8)
                
                // 提示：若某时段负面突增
                if let spike = segments.first(where: { $0.negativeCount >= 3 && $0.negativeCount > $0.positiveCount }) {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "lightbulb.fill")
                            .foregroundColor(.orange)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("建议关注")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                            Text("\(spike.label) 附近负面评论较多（\(spike.negativeCount) 条），可回顾该段内容是否需优化。")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(10)
                }
                
                Spacer(minLength: 40)
            }
            .padding()
        }
    }
    
    private func legendItem(color: Color, text: String) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 10, height: 10)
            Text(text)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private func loadEpisodeData() {
        isLoading = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            let raw = Self.sampleEpisodeCommentsWithTimestamps()
            let classified = raw.map { comment in
                var c = comment
                c.category = classifier?.classify(comment.content) ?? .neutral
                return c
            }
            self.episodeComments = classified
            recomputeSegments(segmentLength: segmentChoice.seconds)
            isLoading = false
        }
    }
    
    private func recomputeSegments(segmentLength: TimeInterval) {
        let segmentCount = max(1, Int(ceil(episodeDuration / segmentLength)))
        var newSegments: [SentimentSegment] = (0..<segmentCount).map { i in
            let start = TimeInterval(i) * segmentLength
            let end = min(start + segmentLength, episodeDuration)
            let label = Self.formatTime(start)
            return SentimentSegment(
                segmentIndex: i,
                label: label,
                startSeconds: start,
                endSeconds: end,
                positiveCount: 0,
                negativeCount: 0,
                neutralCount: 0
            )
        }
        
        for comment in episodeComments {
            guard let t = comment.timestampInEpisode,
                  let cat = comment.category else { continue }
            let index = min(Int(t / segmentLength), newSegments.count - 1)
            if index >= 0 {
                var seg = newSegments[index]
                switch cat {
                case .positive: seg.positiveCount += 1
                case .negative: seg.negativeCount += 1
                case .neutral: seg.neutralCount += 1
                }
                newSegments[index] = seg
            }
        }
        
        self.segments = newSegments
    }
    
    private static func formatTime(_ seconds: TimeInterval) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return String(format: "%d:%02d", m, s)
    }
    
    /// 示例：一集 60 分钟节目的带时间戳评论（模拟第 15 分钟附近负面增多）
    private static func sampleEpisodeCommentsWithTimestamps() -> [Comment] {
        let now = Date()
        return [
            Comment(content: "开头就很抓人，期待后面！", author: "A", date: now, timestampInEpisode: 60),
            Comment(content: "主播声音好听，节奏舒服。", author: "B", date: now, timestampInEpisode: 180),
            Comment(content: "这段讲得很清楚，有收获。", author: "C", date: now, timestampInEpisode: 420),
            Comment(content: "这里讲得太快了，没跟上。", author: "D", date: now, timestampInEpisode: 14 * 60),
            Comment(content: "15分钟这段有点水，希望后面更干货。", author: "E", date: now, timestampInEpisode: 15 * 60),
            Comment(content: "刚刚那段逻辑有点乱，听懵了。", author: "F", date: now, timestampInEpisode: 16 * 60),
            Comment(content: "广告插得有点突兀。", author: "G", date: now, timestampInEpisode: 18 * 60),
            Comment(content: "广告之后内容又回来了，不错。", author: "H", date: now, timestampInEpisode: 22 * 60),
            Comment(content: "中段开始渐入佳境。", author: "I", date: now, timestampInEpisode: 25 * 60),
            Comment(content: "这个观点很有启发！", author: "J", date: now, timestampInEpisode: 32 * 60),
            Comment(content: "例子举得好，容易理解。", author: "K", date: now, timestampInEpisode: 35 * 60),
            Comment(content: "后面节奏又有点拖。", author: "L", date: now, timestampInEpisode: 48 * 60),
            Comment(content: "结尾总结到位，整体满意。", author: "M", date: now, timestampInEpisode: 55 * 60),
            Comment(content: "整期质量不错，会推荐给朋友。", author: "N", date: now, timestampInEpisode: 58 * 60),
        ]
    }
}

#Preview {
    CurveTabView()
}
