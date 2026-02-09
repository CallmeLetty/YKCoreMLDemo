//
//  TextClassifierApp.swift
//  TextClassifier
//
//  Created by Yakamoz on 2026/2/7.
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
            ContentView()
                .tabItem {
                    Label("文本分类", systemImage: "doc.text.magnifyingglass")
                }
            
            SecondTabView()
                .tabItem {
                    Label("历史记录", systemImage: "clock.arrow.circlepath")
                }
            
            ThirdTabView()
                .tabItem {
                    Label("设置", systemImage: "gearshape")
                }
        }
    }
}

// 第二个Tab - 历史记录
struct SecondTabView: View {
    var body: some View {
        VStack {
            Text("历史记录")
                .font(.title)
                .fontWeight(.bold)
                .padding()
            
            Text("这里可以显示分类历史")
                .foregroundColor(.gray)
            
            Spacer()
        }
        .padding()
    }
}

// 第三个Tab - 设置
struct ThirdTabView: View {
    var body: some View {
        VStack {
            Text("设置")
                .font(.title)
                .fontWeight(.bold)
                .padding()
            
            Text("这里可以配置模型参数")
                .foregroundColor(.gray)
            
            Spacer()
        }
        .padding()
    }
}
