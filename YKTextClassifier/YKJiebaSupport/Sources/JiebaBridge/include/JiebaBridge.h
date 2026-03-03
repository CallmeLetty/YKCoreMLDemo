#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface JiebaBridge : NSObject

+ (instancetype)shared;
- (void)setup;
- (NSArray<NSString *> *)cut:(NSString *)text useHMM:(BOOL)useHMM;

/// 从文本中提取关键词（基于 IDF + 词频），用于如「负面评论痛点」等场景。topN 建议 5~15。
- (NSArray<NSString *> *)extractKeywords:(NSString *)text topN:(NSInteger)topN;

@end

NS_ASSUME_NONNULL_END

