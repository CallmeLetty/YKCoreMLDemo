//
//  YKTextClassifierApp.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//

import SwiftUI
import YKJiebaSupport
import FoundationModels

@main
struct TextClassifierApp: App {
    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
    }
}

struct MainTabView: View {
    var body: some View {
        TabView {
            // Tab 1: 文本感情分类（内含顶部三个子 Tab）
            SentimentTabView()
                .tabItem {
                    Label("文本感情分类", systemImage: "doc.text.magnifyingglass")
                }

            // Tab 2: 评论分类
            CommentsListView()
                .tabItem {
                    Label("评论分类", systemImage: "bubble.left.and.bubble.right")
                }
            
            // Tab 3: 情感曲线图
            CurveTabView()
                .tabItem {
                    Label("情感曲线图", systemImage: "chart.line.uptrend.xyaxis")
                }
        }
        .onAppear {
             Task {
                 do {
                     let req = CommentListPrimaryRequest(
                         targetId: "episode_id_xxx",
                         order: "CREATED_AT_DESC",
                         locatedId: nil,
                         loadMoreKey: nil
                     )
                     let (comments, loadMoreKey, totalCount, notFoundToastText) = try await requestCommentListPrimary(req)
                     print(comments.count, totalCount ?? 0)
                 } catch {
                     print(error)
                 }
             }
        }
    }
}
