//
//  AppTheme.swift
//  YKTextClassifier
//
//  统一主题与动效，让 UI 更高级、酷炫
//

import SwiftUI

// MARK: - 日/夜间主题 Environment
private struct IsDarkModeKey: EnvironmentKey {
    static let defaultValue = true
}
extension EnvironmentValues {
    var isDarkMode: Bool {
        get { self[IsDarkModeKey.self] }
        set { self[IsDarkModeKey.self] = newValue }
    }
}

// MARK: - 渐变色与主题色（支持日/夜间）
enum AppTheme {
    static let gradientStart = Color(red: 0.15, green: 0.12, blue: 0.28)
    static let gradientEnd = Color(red: 0.08, green: 0.06, blue: 0.18)
    static var backgroundGradient: LinearGradient {
        backgroundGradient(dark: true)
    }
    
    /// 根据日/夜间返回背景渐变
    static func backgroundGradient(dark: Bool) -> LinearGradient {
        if dark {
            return LinearGradient(
                colors: [gradientStart, gradientEnd],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        }
        return LinearGradient(
            colors: [Color(red: 0.95, green: 0.94, blue: 0.98), Color(red: 0.88, green: 0.86, blue: 0.95)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
    
    static let accentGradient = LinearGradient(
        colors: [Color(red: 0.4, green: 0.35, blue: 1.0), Color(red: 0.6, green: 0.4, blue: 1.0)],
        startPoint: .leading,
        endPoint: .trailing
    )
    
    static let cardBackground = Color.white.opacity(0.08)
    static let cardBorder = Color.white.opacity(0.12)
    static let glassOpacity: Double = 0.12
    
    /// 主文字色（随主题）
    static func textPrimary(dark: Bool) -> Color {
        dark ? Color.white.opacity(0.95) : Color.primary
    }
    
    static func textSecondary(dark: Bool) -> Color {
        dark ? Color.white.opacity(0.7) : Color.secondary
 }
    
    static func cardStrokeColor(dark: Bool) -> Color {
        dark ? Color.white.opacity(0.12) : Color.black.opacity(0.08)
    }
}

// MARK: - 动效常量
enum AppAnimation {
    static let springQuick = Animation.spring(response: 0.35, dampingFraction: 0.75)
    static let springBouncy = Animation.spring(response: 0.45, dampingFraction: 0.7)
    static let springSmooth = Animation.spring(response: 0.5, dampingFraction: 0.85)
    static let easeOutShort = Animation.easeOut(duration: 0.25)
    static let staggerDelay: Double = 0.04
}

// MARK: - 视图扩展：入场 / 交互动效
extension View {
    /// 带延迟的渐显 + 轻微上移，用于列表项等
    func staggeredAppear(index: Int) -> some View {
        modifier(StaggeredAppearModifier(index: index))
    }
    
    /// 按压缩放反馈
    func pressScaleEffect(isPressed: Bool) -> some View {
        scaleEffect(isPressed ? 0.97 : 1.0)
        .animation(AppAnimation.springQuick, value: isPressed)
    }
    
    /// 卡片式背景（毛玻璃 + 圆角描边）
    func cardStyle() -> some View {
        self
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(AppTheme.cardBorder, lineWidth: 1)
                    )
            )
    }
}

struct StaggeredAppearModifier: ViewModifier {
    let index: Int
    @State private var appeared = false
    
    func body(content: Content) -> some View {
        content
            .opacity(appeared ? 1 : 0)
            .offset(y: appeared ? 0 : 12)
            .onAppear {
                withAnimation(AppAnimation.springSmooth.delay(Double(index) * AppAnimation.staggerDelay)) {
                    appeared = true
                }
            }
    }
}
