//
//  CommentModel.swift
//  YKTextClassifier
//
//  Created by YakaLiu on 2026/3/6.
//

//import DifferenceKit
//import Foundation
//import ObjectMapper
//
//public struct CollectRelatedInfo: Mappable {
//    public init?(map: ObjectMapper.Map) {
//        do {
//            episode = try map.value("episode")
//        } catch {
//            return nil
//        }
//    }
//
//    public mutating func mapping(map: ObjectMapper.Map) {}
//
//    public var episode: Episode?
//}
//
//public struct Comment: Mappable {
//    public let id: String
//    public var author: User?
//    public var text: String = ""
//    public var level: Int = 1
//    public var threadReplyCount: Int = 0
//    public var likeCount: Int = 0
//    public var liked: Bool = false
//    public var pinned: Bool = false
//    public var createdAt: Date?
//    public var pid: String = "" // Podcast
//    public var ownerId: String = "" // Episode
//    public var ownerType: String = ""
//    public var thread: String = "" // Level 1 Comment
//    public var replyToCommentId: String? // (level - 1) Comment
//    public var replyToUser: User?
//    public var replyToCommentText: String?
//    public var replyToCommentAuthorAssociation: AuthorAssociation = .none
//    public var replies: [Comment] = []
//    public var status: Status = .normal
//    public var authorAssociation: AuthorAssociation = .none
//    public var podcastAssociation: PodcastAssociation = .none
//    public var permissions: [Permission] = []
//    public var isAuthorMuted = false
//    public var entities: [EntityParseResult] = []
//    public var badges: [Badge] = []
//    public var voice: Voice? // v2.22
//    public var collected: Bool = false
//    public var collectInfo: CollectRelatedInfo?
//    public var ipLoc: String?
//    public var isFriendly: Bool = false
//
//    public init?(map: Map) {
//        do {
//            id = try map.value("id")
//            collectInfo = CollectRelatedInfo(map: map)
//        } catch {
//            return nil
//        }
//    }
//
//    public mutating func mapping(map: Map) {
//        id >>> map["id"]
//        author <- map["author"]
//        text <- map["text"]
//        level <- map["level"]
//        threadReplyCount <- map["threadReplyCount"]
//        likeCount <- map["likeCount"]
//        liked <- map["liked"]
//        pinned <- map["pinned"]
//        createdAt <- (map["createdAt"], DateTransform())
//        pid <- map["pid"]
//        ownerId <- map["owner.id"]
//        ownerType <- map["owner.type"]
//        thread <- map["thread"]
//        replyToCommentId <- map["replyToComment.id"]
//        replyToUser <- map["replyToComment.author"]
//        replyToCommentText <- map["replyToComment.text"]
//        replyToCommentAuthorAssociation <- map["replyToComment.authorAssociation"]
//        replies <- map["replies"]
//        status <- map["status"]
//        authorAssociation <- map["authorAssociation"]
//        permissions <- map["permissions"]
//        isAuthorMuted <- map["isAuthorMuted"]
//        entities <- map["entities"]
//        badges <- map["badges"]
//        voice <- map["voice"]
//        collected <- map["collected"]
//        ipLoc <- map["ipLoc"]
//        isFriendly <- map["isFriendly"]
//        podcastAssociation <- map["podcastAssociation"]
//    }
//
//    public var commentPermissionStatus: Permission.Status {
//        return permissions.getStatus(of: .comment)
//    }
//
//    public var sharePermissionStatus: Permission.Status {
//        return permissions.getStatus(of: .share)
//    }
//
//    public var pinPermissionStatus: Permission.Status {
//        return permissions.getStatus(of: .commentPinOperation, defaultStatus: .denied)
//    }
//
//    public var getDeletePermissionStatus: Permission.Status {
//        return permissions.getStatus(of: .delete, defaultStatus: .denied)
//    }
//
//    public var muteAuthorPermissionStatus: Permission.Status {
//        return permissions.getStatus(of: .muteCommentAuthor, defaultStatus: .denied)
//    }
//}
//
//extension Comment: Differentiable {
//    public var differenceIdentifier: String {
//        return id
//    }
//
//    public func isContentEqual(to source: Comment) -> Bool {
//        return differenceIdentifier == source.id &&
//            threadReplyCount == source.threadReplyCount &&
//            likeCount == source.likeCount &&
//            liked == source.liked &&
//            pinned == source.pinned &&
//            replies.map { $0.id } == source.replies.map { $0.id } &&
//            author.isContentEqual(to: source.author)
//    }
//}
//
//public extension Comment {
//    enum Status: String {
//        case normal = "NORMAL"
//        case removed = "REMOVED"
//    }
//
//    enum AuthorAssociation: String {
//        case none = "NONE"
//        case podcaster = "PODCASTER"
//    }
//
//    enum PodcastAssociation: String {
//        case none = "NONE"
//        case original = "ORIGINAL"
//        case coCreator = "CO_CREATOR"
//        case brand = "BRAND"
//
//        public var title: String? {
//            switch self {
//            case .none:
//                nil
//            case .original:
//                "原创"
//            case .coCreator:
//                "联创"
//            case .brand:
//                "品牌"
//            }
//        }
//    }
//
//    struct Badge: Mappable {
//        public struct Icon: Mappable {
//            public let height: Int
//            public let width: Int
//            public let url: String
//            public let format: String
//
//            public init?(map: Map) {
//                do {
//                    height = try map.value("height")
//                    width = try map.value("width")
//                    format = try map.value("format")
//                    url = try map.value("picUrl")
//                } catch {
//                    return nil
//                }
//            }
//
//            public mutating func mapping(map: Map) {}
//        }
//
//        public let icon: Icon
//        public var tip: String = ""
//
//        public init?(map: Map) {
//            do {
//                icon = try map.value("icon")
//            } catch {
//                return nil
//            }
//        }
//
//        public mutating func mapping(map: Map) {
//            tip <- map["tip"]
//        }
//    }
//
//    struct Voice: Mappable {
//        public let url: URL
//        public let duration: Double
//        public let waveform: [Int]
//
//        public init?(map: Map) {
//            do {
//                waveform = try map.value("waveform")
//                url = try map.value("url", using: URLEncodeTransform())
//                duration = try map.value("duration")
//            } catch {
//                return nil
//            }
//        }
//
//        public mutating func mapping(map: Map) {
//            waveform >>> map["waveform"]
//            url >>> (map["url"], URLEncodeTransform())
//            duration >>> map["duration"]
//        }
//    }
//
//    struct EntityParseResult: Mappable {
//        public enum EntityType: String {
//            case podcast = "PODCAST"
//            case episode = "EPISODE"
//        }
//
//        public let origin: String
//        public var type: EntityType
//
//        public var podcast: Podcast?
//        public var episode: Episode?
//
//        public var commentLinkTitleOrUrl: String {
//            switch type {
//            case .episode:
//                return episode?.title ?? self.origin
//            case .podcast:
//                if let podcastTitle = podcast?.title {
//                    return "《\(podcastTitle)》"
//                } else {
//                    return origin
//                }
//            }
//        }
//
//        public var customSchemeURL: URL? {
//            switch type {
//            case .episode:
//                return URL(string: "cosmos://page.cos/shownotes/\(episode?.id ?? "")")
//            case .podcast:
//                return URL(string: "cosmos://page.cos/podcast/\(podcast?.id ?? "")")
//            }
//        }
//
//        public init?(map: Map) {
//            do {
//                origin = try map.value("origin")
//                type = try map.value("entity.type")
//            } catch {
//                return nil
//            }
//        }
//
//        public mutating func mapping(map: Map) {
//            origin >>> map["origin"]
//            switch type {
//            case .episode:
//                episode <- map["entity"]
//            case .podcast:
//                podcast <- map["entity"]
//            }
//            type >>> map["entity.type"]
//        }
//    }
//}
//
//public struct TimestampPrimary: Mappable {
//    public let id: String
//    public let type: String
//    public let timestamp: Double
//
//    public init?(map: Map) {
//        do {
//            id = try map.value("id")
//            type = try map.value("type")
//            timestamp = try map.value("timestamp")
//        } catch {
//            return nil
//        }
//    }
//
//    public mutating func mapping(map: Map) {}
//}
//
//extension TimestampPrimary: Equatable {
//    public static func == (lhs: TimestampPrimary, rhs: TimestampPrimary) -> Bool {
//        return lhs.id == rhs.id
//    }
//}
