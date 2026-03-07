//
//  JiebaBridge.m
//  YKJiebaSupport
//
//  Created by Yakamoz on 2026/2/16.
//


#import "include/JiebaBridge.h"

// 引入 C++ 头文件
// 注意：由于我们在 Package.swift 设置了 search path，这里可以直接引用
#include "cppjieba/Jieba.hpp" 

using namespace std;

@implementation JiebaBridge {
    cppjieba::Jieba *_jieba;
}

+ (instancetype)shared {
    static JiebaBridge *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[JiebaBridge alloc] init];
    });
    return instance;
}

- (void)setup {
    if (_jieba) {
        return;
    }
    static dispatch_once_t onceToken;
    static NSLock *setupLock = nil;
    dispatch_once(&onceToken, ^{
        setupLock = [[NSLock alloc] init];
    });
    [setupLock lock];
    if (_jieba) {
        [setupLock unlock];
        return;
    }
    // 在 Swift Package 中，资源文件会被打包到 YKJiebaSupport_JiebaBridge.bundle 中
    NSBundle *resourceBundle = nil;
    NSString *dictRoot = nil;
    
    // 方法1: 查找 YKJiebaSupport_JiebaBridge.bundle（SPM 生成的资源 bundle）
    NSString *bundlePath = [[NSBundle mainBundle] pathForResource:@"YKJiebaSupport_JiebaBridge" ofType:@"bundle"];
    if (bundlePath) {
        resourceBundle = [NSBundle bundleWithPath:bundlePath];
        dictRoot = [resourceBundle pathForResource:@"dict" ofType:nil];
    }
    
    // 方法2: 遍历所有 bundle 查找包含 dict 资源的 bundle
    if (!dictRoot) {
        for (NSBundle *bundle in [NSBundle allBundles]) {
            NSString *path = [bundle pathForResource:@"dict" ofType:nil];
            if (path) {
                dictRoot = path;
                resourceBundle = bundle;
                break;
            }
        }
    }
    
    // 方法3: 尝试从主 bundle 中查找
    if (!dictRoot) {
        dictRoot = [[NSBundle mainBundle] pathForResource:@"dict" ofType:nil];
    }
    
    if (!dictRoot) {
        NSLog(@"[JiebaError] 找不到 dict 文件夹！");
        NSLog(@"[JiebaError] 主 bundle 路径: %@", [[NSBundle mainBundle] bundlePath]);
        NSLog(@"[JiebaError] 已尝试的 bundle:");
        for (NSBundle *bundle in [NSBundle allBundles]) {
            NSLog(@"  - %@", bundle.bundleIdentifier ?: bundle.bundlePath);
        }
        [setupLock unlock];
        return;
    }
    
//    NSLog(@"[Jieba] 找到字典路径: %@", dictRoot);
    
    string root = [dictRoot UTF8String];
    
    // 拼接路径
    string dictPath      = root + "/jieba.dict.utf8";
    string hmmPath       = root + "/hmm_model.utf8";
    string userDictPath  = root + "/user.dict.utf8";
    string idfPath       = root + "/idf.utf8";
    string stopWordPath  = root + "/stop_words.utf8";
    
    // 验证文件是否存在
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL dictExists = [fm fileExistsAtPath:[NSString stringWithUTF8String:dictPath.c_str()]];
    BOOL hmmExists = [fm fileExistsAtPath:[NSString stringWithUTF8String:hmmPath.c_str()]];
    
    NSLog(@"[Jieba] 字典文件检查:");
    NSLog(@"  - jieba.dict.utf8: %@", dictExists ? @"✓" : @"✗");
    NSLog(@"  - hmm_model.utf8: %@", hmmExists ? @"✓" : @"✗");
    
    if (!dictExists) {
        NSLog(@"[JiebaError] 主字典文件不存在: %s", dictPath.c_str());
        [setupLock unlock];
        return;
    }
    
    try {
        _jieba = new cppjieba::Jieba(dictPath, hmmPath, userDictPath, idfPath, stopWordPath);
        NSLog(@"[Jieba] ✅ 初始化成功");
    } catch (const std::exception& e) {
        NSLog(@"[JiebaError] 初始化失败: %s", e.what());
    } catch (...) {
        NSLog(@"[JiebaError] 初始化崩溃，未知错误");
    }
    [setupLock unlock];
}

- (NSArray<NSString *> *)cut:(NSString *)text useHMM:(BOOL)useHMM {
    if (!_jieba || !text) return @[];
    
    string sentence = [text UTF8String];
    vector<string> words;
    
    _jieba->Cut(sentence, words, useHMM);
    
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:words.size()];
    for (const auto& word : words) {
        [result addObject:[NSString stringWithUTF8String:word.c_str()]];
    }
    
    return [result copy];
}

- (NSArray<NSString *> *)extractKeywords:(NSString *)text topN:(NSInteger)topN {
    if (!_jieba || !text || text.length == 0 || topN <= 0) return @[];
    
    string sentence = [text UTF8String];
    std::vector<std::pair<std::string, double>> kws;
    _jieba->extractor.Extract(sentence, kws, (size_t)topN);
    
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:kws.size()];
    for (const auto& p : kws) {
        [result addObject:[NSString stringWithUTF8String:p.first.c_str()]];
    }
    return [result copy];
}

- (NSArray<NSDictionary<NSString *, id> *> *)extractKeywordsWithWeights:(NSString *)text topN:(NSInteger)topN {
    if (!_jieba || !text || text.length == 0 || topN <= 0) return @[];
    
    string sentence = [text UTF8String];
    std::vector<std::pair<std::string, double>> kws;
    _jieba->extractor.Extract(sentence, kws, (size_t)topN);
    
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:kws.size()];
    for (const auto& p : kws) {
        [result addObject:@{
            @"word": [NSString stringWithUTF8String:p.first.c_str()],
            @"weight": @(p.second)
        }];
    }
    return [result copy];
}

- (NSArray<NSString *> *)extractKeywordsTFIDFFromTokenizedDocuments:(NSArray<NSArray<NSString *> *> *)documents
                                                              topN:(NSInteger)topN
                                                     minWordLength:(NSInteger)minWordLength {
    if (!documents || documents.count == 0 || topN <= 0) return @[];
    
    NSInteger N = documents.count;
    NSMutableDictionary<NSString *, NSNumber *> *tf = [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSNumber *> *df = [NSMutableDictionary dictionary];
    
    for (NSArray<NSString *> *tokens in documents) {
        if (![tokens isKindOfClass:[NSArray class]] || tokens.count == 0) continue;
        NSMutableSet<NSString *> *seenInDoc = [NSMutableSet set];
        for (NSString *word in tokens) {
            if (![word isKindOfClass:[NSString class]] || (NSInteger)word.length < minWordLength) continue;
            NSString *w = [word copy];
            tf[w] = @(tf[w].integerValue + 1);
            [seenInDoc addObject:w];
        }
        for (NSString *w in seenInDoc) {
            df[w] = @(df[w].integerValue + 1);
        }
    }
    
    NSMutableArray<NSDictionary *> *scored = [NSMutableArray arrayWithCapacity:tf.count];
    for (NSString *word in tf) {
        double tfVal = tf[word].doubleValue;
        double dfVal = df[word].doubleValue + 1.0;
        double idf = log((double)(N + 1) / dfVal);
        [scored addObject:@{ @"word": word, @"score": @(tfVal * idf) }];
    }
    [scored sortUsingComparator:^NSComparisonResult(NSDictionary *a, NSDictionary *b) {
        return [b[@"score"] compare:a[@"score"]];
    }];
    
    NSMutableArray *result = [NSMutableArray arrayWithCapacity:MIN((NSUInteger)topN, scored.count)];
    for (NSUInteger i = 0; i < (NSUInteger)topN && i < scored.count; i++) {
        [result addObject:scored[i][@"word"]];
    }
    return [result copy];
}

- (void)dealloc {
    if (_jieba) delete _jieba;
}

@end
