//
//  VocabularyManager.swift
//  YKTextClassifier
//
//  Created by Yakamoz on 2026/2/17.
//

import Foundation
import YKJiebaSupport // 导入你的本地包

/// 是否在分词前去掉标点（与 Python 训练脚本 model_def.STRIP_PUNCTUATION 保持一致）
private let stripPunctuation = true

/// 分词前预处理：去掉中文/英文标点与空白，与 Python _normalize_text 对齐
private func normalizeForTokenize(_ text: String) -> String {
    guard stripPunctuation, !text.isEmpty else { return text }
    return text.unicodeScalars
        .filter { scalar in
            let u = scalar.value
            if u <= 0x20 || (u >= 0x7f && u <= 0xa0) { return false } // 空白、控制字符
            if u >= 0x3000 && u <= 0x303f { return false } // CJK 符号和标点
            if u >= 0xff00 && u <= 0xffef { return false } // 全角
            if u >= 0x2000 && u <= 0x206f { return false } // 通用标点等
            switch u {
            case 0x21...0x2f, 0x3a...0x40, 0x5b...0x60, 0x7b...0x7e: return false
            default: return true
            }
        }
        .map { Character($0) }
        .reduce("") { $0 + String($1) }
}

class VocabularyManager {
    private var isInitialized = false
    
    func initJieba() {
        guard !isInitialized else {
            print("[VocabularyManager] Jieba 已经初始化")
            return
        }
        
        print("[VocabularyManager] 开始初始化 Jieba...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            JiebaBridge.shared().setup()
            self?.isInitialized = true
            print("[VocabularyManager] Jieba 初始化完成")
        }
    }
    
    func tokenize(_ text: String) -> [String] {
        if !isInitialized {
            print("[VocabularyManager] ⚠️ 警告: Jieba 未初始化，正在初始化...")
            // 同步初始化（不推荐，但确保能用）
            JiebaBridge.shared().setup()
            isInitialized = true
        }
        
        let normalized = normalizeForTokenize(text)
        let result = JiebaBridge.shared().cut(normalized, useHMM: true)
        
        if result.isEmpty && !text.isEmpty {
            print("[VocabularyManager] ⚠️ 分词结果为空，输入: \(text)")
        }
        
        return result
    }
}
