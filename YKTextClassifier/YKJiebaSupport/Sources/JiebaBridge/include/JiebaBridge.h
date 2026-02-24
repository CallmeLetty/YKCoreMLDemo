#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface JiebaBridge : NSObject

+ (instancetype)shared;
- (void)setup;
- (NSArray<NSString *> *)cut:(NSString *)text useHMM:(BOOL)useHMM;

@end

NS_ASSUME_NONNULL_END

