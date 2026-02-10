//
//  ThirdTabView.swift
//  YKTextClassifier
//
//  Created by Yakamoz on 2026/2/10.
//

import SwiftUI

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
