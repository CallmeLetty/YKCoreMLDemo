//
//  YKTextClassifierApp.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/2/9.
//

import SwiftUI

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
            FirstTabView()
                .tabItem {
                    Label("文本分类", systemImage: "doc.text.magnifyingglass")
                }

            SecondTabView()
                .tabItem {
                    Label("历史记录", systemImage: "clock.arrow.circlepath")
                }

            PyContentView()
                .tabItem {
                    Label("设置", systemImage: "gearshape")
                }
        }
    }
}
