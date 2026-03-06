//
//  CommentListPrimaryRequest.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/3/6.
//


import Foundation

// MARK: - 请求参数
struct CommentListPrimaryRequest {
    let targetId: String   // 单集 id (owner.id)
    let order: String      // 排序，如 "CREATED_AT_DESC"
    let locatedId: String?
    let loadMoreKey: Any?  // 分页游标，如 String 或服务端返回的任意类型
}

// MARK: - 响应模型（按你接口实际字段调整）
struct CommentListPrimaryResponse: Decodable {
    let data: [CommentDTO]?
    let loadMoreKey: String?
    let totalCount: Int?
    let notFoundToastText: String?
    
    enum CodingKeys: String, CodingKey {
        case data
        case loadMoreKey
        case totalCount
        case notFoundToastText
    }
}

// 若接口返回的评论结构不同，可只保留你需要的字段
struct CommentDTO: Decodable {
    let id: String
    let text: String?
    let level: Int?
    let threadReplyCount: Int?
    let likeCount: Int?
    let liked: Bool?
    let pinned: Bool?
    let createdAt: String?
    let pid: String?
    let ownerId: String?
    let ownerType: String?
    let thread: String?
    let replyToCommentId: String?
    let author: UserDTO?
    let replyToUser: UserDTO?
    let replyToCommentText: String?
    let replies: [CommentDTO]?
    let status: String?
    // 其他字段按需添加
}

struct UserDTO: Decodable {
    let id: String?
    let nickname: String?
    let avatar: String?
    // 其他字段按需
}

// MARK: - URLSession 调用
func requestCommentListPrimary(
    _ req: CommentListPrimaryRequest,
    baseURL: String = "https://api.xiaoyuzhoufm.com",
    path: String = "/v1/comment/list-primary",
    authorization: String? = nil
) async throws -> (comments: [CommentDTO], loadMoreKey: String?, totalCount: Int?, notFoundToastText: String?) {
    let url = URL(string: baseURL + path)!
    var urlRequest = URLRequest(url: url)
    urlRequest.httpMethod = "POST"
    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let auth = authorization {
        urlRequest.setValue(auth, forHTTPHeaderField: "Authorization")
    }
    
    var body: [String: Any] = [
        "owner": ["id": req.targetId, "type": "EPISODE"],
        "order": req.order
    ]
    if let locatedId = req.locatedId {
        body["locatedId"] = locatedId
    }
    if let key = req.loadMoreKey {
        body["loadMoreKey"] = key
    }
    
    urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)
    
    let (data, response) = try await URLSession.shared.data(for: urlRequest)
    
    guard let http = response as? HTTPURLResponse else {
        throw URLError(.unknown)
    }
    guard (200...299).contains(http.statusCode) else {
        throw URLError(.init(rawValue: http.statusCode))
    }
    
    let decoded = try JSONDecoder().decode(CommentListPrimaryResponse.self, from: data)
    return (
        decoded.data ?? [],
        decoded.loadMoreKey,
        decoded.totalCount,
        decoded.notFoundToastText
    )
}

// MARK: - 使用示例（async/await）
// Task {
//     do {
//         let req = CommentListPrimaryRequest(
//             targetId: "episode_id_xxx",
//             order: "CREATED_AT_DESC",
//             locatedId: nil,
//             loadMoreKey: nil
//         )
//         let (comments, loadMoreKey, totalCount, notFoundToastText) = try await requestCommentListPrimary(req)
//         print(comments.count, totalCount ?? 0)
//     } catch {
//         print(error)
//     }
// }
//Task {
//    do {
//        let (arrayWithLoadMore, totalCount, notFoundToastText) = try await CommentApi.requestCommentListPrimary(
//            targetId: "69aa35a27503cce990c71479",
//            order: "HOT",
//            locatedId: nil,
//            loadMoreKey: nil
//        )
//        let comments = arrayWithLoadMore.array
//        // 使用 comments, totalCount, notFoundToastText
//    } catch {
//        // 处理错误（含 401 等）
//    }
//}
//@@ -8,14 +8,107 @@
//
//import AluminumKit
//import Combine
//import Foundation
//import JKNetworking
//import ObjectMapper
//import RxSwift
//import UIKit
//import YZAppGroupKit
//import YZFactoryKit
//import YZNetworkingKit
//
//enum CommentApi {
//    static let commentLikeUpdated = PassthroughSubject<Comment, Never>()
//
//    /// URLSession 实现的一级评论列表请求，请求头与 curl/App 一致，避免 401
//    static func requestCommentListPrimary(
//        targetId: String,
//        order: String = "HOT",
//        locatedId: String? = nil,
//        loadMoreKey: Any? = nil
//    ) async throws -> (ArrayWithLoadMore<Comment>, Int, String?) {
//        let baseURL = ApiEnvironment.current.apiUrl + "/v1/comment/list-primary"
//        guard let url = URL(string: baseURL) else { throw URLError(.badURL) }
//
//        var urlRequest = URLRequest(url: url)
//        urlRequest.httpMethod = "POST"
//        urlRequest.setValue("application/json", forHTTPHeaderField: "content-type")
//
//        // 鉴权（必带，否则 401）
//        let provider = GlobalContainer.networkProvider()
//        for (key, value) in provider.accessTokenHeader {
//            urlRequest.setValue(value, forHTTPHeaderField: key)
//        }
//
//        // 与 curl / HeaderPlugin 一致的请求头
//        let localTimeFormatter: DateFormatter = {
//            let f = DateFormatter()
//            f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
//            f.locale = Locale(identifier: "en_US")
//            f.timeZone = TimeZone(abbreviation: "UTC")
//            return f
//        }()
//
//        var headers: [String: String] = [
//            "user-agent": userAgent,
//            "market": "AppStore",
//            "x-jike-device-properties": DeviceUtil.idfaAndIdfvJSON,
//            "app-buildno": DeviceUtil.buildNo,
//            "x-jike-device-id": deviceId,
//            "local-time": localTimeFormatter.string(from: Date()),
//            "os": "ios",
//            "x-custom-xiaoyuzhou-app-dev": ApiEnvironment.currentBetaEnvironmentOption?.value ?? "",
//            "manufacturer": "Apple",
//            "bundleid": DeviceUtil.bundleId,
//            "accept-language": Locale.preferredLanguages.first ?? "zh-Hans-CN;q=1.0",
//            "timezone": NSTimeZone.local.identifier,
//            "model": UIDevice.current.modelName,
//            "app-permissions": AuthorizationService.authorizationValueForRequestHeader,
//            "accept": "*/*",
//            "app-version": DeviceUtil.appVersion,
//            "wificonnected": GlobalContainer.networkStatus().value == .wifi ? "true" : "false",
//            "os-version": DeviceUtil.osVersionForHeader,
//        ]
//        if let abTest = GlobalContainer.userContext().abTestInfo {
//            headers["abtest-info"] = abTest
//        }
//        for (key, value) in httpStaticHeaders {
//            headers[key.lowercased()] = value
//        }
//        for (key, value) in headers {
//            urlRequest.setValue(value, forHTTPHeaderField: key)
//        }
//
//        var body: [String: Any] = [
//            "owner": ["id": targetId, "type": "EPISODE"],
//            "order": order,
//        ]
//        if let locatedId = locatedId { body["locatedId"] = locatedId }
//        if let key = loadMoreKey { body["loadMoreKey"] = key }
//        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)
//
//        let (data, response) = try await URLSession.shared.data(for: urlRequest)
//        guard let http = response as? HTTPURLResponse else { throw URLError(.unknown) }
//        guard (200 ... 299).contains(http.statusCode) else {
//            throw URLError(.init(rawValue: http.statusCode))
//        }
//
//        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
//            throw URLError(.cannotParseResponse)
//        }
//        let dataArray = json["data"] as? [[String: Any]] ?? []
//        let comments = Mapper<Comment>().mapArray(JSONObject: dataArray) ?? []
//        let totalCount = json["totalCount"] as? Int ?? 0
//        let notFoundToastText = json["notFoundToastText"] as? String
//        let nextKey = json["loadMoreKey"] ?? json["loadNextKey"]
//        let prevKey = json["loadPrevKey"]
//        let loadMoreKeyValue: Any? = (nextKey != nil || prevKey != nil)
//            ? LoadMoreContext(nextKey: nextKey, prevKey: prevKey)
//            : nil
//        let arrayWithLoadMore = ArrayWithLoadMore(array: comments, loadMoreKey: loadMoreKeyValue)
//        return (arrayWithLoadMore, totalCount, notFoundToastText)
//    }
//
//    static func get(id: String) -> Single<Comment> {
//        JKRequest.get(path: "comment/get")
//            .addParameters(["commentId": id])
