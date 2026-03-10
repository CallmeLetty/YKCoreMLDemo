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
    @State private var selectedTab: Int = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            // Tab 1: 文本感情分类（内含顶部三个子 Tab）
            SentimentTabView()
                .tabItem {
                    Label("文本感情分类", systemImage: "doc.text.magnifyingglass")
                }
                .tag(0)

            // Tab 2: 评论分类
            CommentsListView()
                .tabItem {
                    Label("评论分类", systemImage: "bubble.left.and.bubble.right")
                }
                .tag(1)
            
            // Tab 3: 情感曲线图
            CurveTabView()
                .tabItem {
                    Label("情感曲线图", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(2)
        }
        .tint(Color(red: 0.55, green: 0.45, blue: 1.0))
        .onAppear {
            // 统一 Tab 栏毛玻璃
            let appearance = UITabBarAppearance()
            appearance.configureWithDefaultBackground()
            UITabBar.appearance().standardAppearance = appearance
            UITabBar.appearance().scrollEdgeAppearance = appearance
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
