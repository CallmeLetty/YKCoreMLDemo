// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "YKJiebaSupport",
    platforms: [.iOS(.v13)], // 根据你的需求调整
    products: [
        .library(
            name: "YKJiebaSupport",
            type: .static,
            targets: ["YKJiebaSupport"]),
    ],
    targets: [
        // Objective-C++ 桥接层（包含 C++ 头文件和字典资源）
        .target(
            name: "JiebaBridge",
            dependencies: [],
            path: "Sources/JiebaBridge",
            exclude: [],
            sources: ["JiebaBridge.mm"],
            resources: [
                .copy("../CppJieba/dict")
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("../CppJieba/include")
            ]
        ),
        
        // Swift 封装层
        .target(
            name: "YKJiebaSupport",
            dependencies: ["JiebaBridge"],
            path: "Sources/YKJiebaSupport"
        )
    ],
    cxxLanguageStandard: .cxx11 // 必须支持 C++11
)
