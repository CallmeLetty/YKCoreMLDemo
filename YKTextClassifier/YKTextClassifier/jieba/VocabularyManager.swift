//
//  VocabularyManager.swift
//  YKTextClassifier
//
//  Created by Yakamoz on 2026/2/17.
//


import YKJiebaSupport // 导入你的本地包

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
        
        let result = JiebaBridge.shared().cut(text, useHMM: true)
        
        if result.isEmpty && !text.isEmpty {
            print("[VocabularyManager] ⚠️ 分词结果为空，输入: \(text)")
        }
        
        return result
    }
}
