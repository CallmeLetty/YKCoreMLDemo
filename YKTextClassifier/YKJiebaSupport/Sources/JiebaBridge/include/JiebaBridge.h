#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface JiebaBridge : NSObject

+ (instancetype)shared;
- (void)setup;

/// 对文本进行中文分词，与 Python 端 jieba.cut 对齐，用于情感分类等场景。调用前需先 setup。
/// @param text 待分词字符串（建议已做去标点预处理，与 Python _normalize_text 一致）
/// @param useHMM 是否使用 HMM 识别未登录词，建议 YES，与 Python 默认行为一致
/// @return 分词结果 token 数组，如 @[@"这期", @"聊", @"的", @"话题", ...]。text 为空或未初始化时返回 @[]
- (NSArray<NSString *> *)cut:(NSString *)text useHMM:(BOOL)useHMM;

/// 从文本中提取关键词（基于 IDF + 词频），用于如「负面评论痛点」等场景。topN 建议 5~15。
- (NSArray<NSString *> *)extractKeywords:(NSString *)text topN:(NSInteger)topN;

/// 提取关键词并返回权重，便于按权重排序或展示。每项为 @{ @"word": NSString, @"weight": NSNumber }。
- (NSArray<NSDictionary<NSString *, id> *> *)extractKeywordsWithWeights:(NSString *)text topN:(NSInteger)topN;

/// 基于已分好词的文档列表做 TF-IDF 关键词提取（与 cut 同源：情感分类用 cut，此处传入 cut 结果即可复用一次分词）。
/// @param documents 每篇文档为分词后的 token 数组，如 [ ["好", "看"], ["很", "好", "看"] ]
/// @param topN 返回前 N 个关键词
/// @param minWordLength 忽略长度小于此值的 token，常用 2
- (NSArray<NSString *> *)extractKeywordsTFIDFFromTokenizedDocuments:(NSArray<NSArray<NSString *> *> *)documents
                                                              topN:(NSInteger)topN
                                                     minWordLength:(NSInteger)minWordLength;

@end

NS_ASSUME_NONNULL_END

