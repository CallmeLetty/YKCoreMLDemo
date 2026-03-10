//
//  SettingsTabView.swift
//  YKTextClassifier
//
//  第 4 个 Tab：设置，含日/夜间模式开关
//

import SwiftUI

struct SettingsTabView: View {
    @Binding var isDarkMode: Bool
    
    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.backgroundGradient(dark: isDarkMode)
                    .ignoresSafeArea()
                
                VStack(spacing: 0) {
                    VStack(spacing: 24) {
                        // 日夜间切换
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("夜间模式")
                                    .font(.headline)
                                    .foregroundColor(AppTheme.textPrimary(dark: isDarkMode))
                                Text("开启后使用深色主题，关闭为浅色主题")
                                    .font(.caption)
                                    .foregroundColor(AppTheme.textSecondary(dark: isDarkMode))
                            }
                            Spacer()
                            Toggle("", isOn: $isDarkMode)
                                .labelsHidden()
                                .tint(Color(red: 0.55, green: 0.45, blue: 1.0))
                        }
                        .padding(20)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(.ultraThinMaterial)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16)
                                        .stroke(AppTheme.cardStrokeColor(dark: isDarkMode), lineWidth: 1)
                                )
                        )
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 24)
                    
                    Spacer()
                }
            }
            .navigationTitle("设置")
            .navigationBarTitleDisplayMode(.large)
            .toolbarColorScheme(isDarkMode ? .dark : .light, for: .navigationBar)
        }
    }
}

#Preview {
    SettingsTabView(isDarkMode: .constant(true))
}
