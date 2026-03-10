//
//  YKTextClassifierApp.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//

import SwiftUI
import UIKit
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
    @AppStorage("isDarkMode") private var isDarkMode = true
    
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
            
            // Tab 4: 设置（日/夜间开关）
            SettingsTabView(isDarkMode: $isDarkMode)
                .tabItem {
                    Label("设置", systemImage: "gearshape.fill")
                }
                .tag(3)
        }
        .environment(\.isDarkMode, isDarkMode)
        .tint(Color(red: 0.55, green: 0.45, blue: 1.0))
        .onAppear {
            applyNavigationBarTitleColor(dark: isDarkMode)
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
        .onChange(of: isDarkMode) { _, newValue in
            applyNavigationBarTitleColor(dark: newValue)
        }
    }
    
    /// 统一设置导航栏大标题与普通标题颜色（日/夜间），避免「评论分类」等标题在暗黑下仍为深色
    private func applyNavigationBarTitleColor(dark: Bool) {
        let titleColor = dark ? UIColor.white : UIColor.darkText
        let navAppearance = UINavigationBarAppearance()
        navAppearance.configureWithTransparentBackground()
        navAppearance.largeTitleTextAttributes = [.foregroundColor: titleColor]
        navAppearance.titleTextAttributes = [.foregroundColor: titleColor]
        UINavigationBar.appearance().standardAppearance = navAppearance
        UINavigationBar.appearance().scrollEdgeAppearance = navAppearance
        UINavigationBar.appearance().compactAppearance = navAppearance
    }
}
