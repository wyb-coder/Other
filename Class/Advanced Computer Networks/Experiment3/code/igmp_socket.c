/**
 * igmp_socket.c - IGMP 报文收发模块
 * 实验三：IGMP协议软件（路由器端）开发
 */

#include "igmp.h"

/**
 * 初始化 IGMP RAW Socket
 */
int igmp_socket_init(const char *ifname)
{
    int sockfd;
    int on = 1;
    unsigned char ttl = IGMP_TTL;
    struct ifreq ifr;

    /* 创建 RAW Socket */
    sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_IGMP);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    /* 设置组播 TTL = 1 */
    if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl)) < 0) {
        perror("setsockopt IP_MULTICAST_TTL");
        close(sockfd);
        return -1;
    }

    /* 允许接收自己发送的组播 */
    if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_LOOP, &on, sizeof(on)) < 0) {
        perror("setsockopt IP_MULTICAST_LOOP");
    }

    /* 绑定到指定接口 */
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ifname, IFNAMSIZ - 1);
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, &ifr, sizeof(ifr)) < 0) {
        perror("setsockopt SO_BINDTODEVICE");
        /* 不是致命错误，继续 */
    }

    /* 获取接口索引 */
    if (ioctl(sockfd, SIOCGIFINDEX, &ifr) < 0) {
        perror("ioctl SIOCGIFINDEX");
    } else {
        g_config.if_index = ifr.ifr_ifindex;
    }

    /* 获取接口 IP 地址 */
    if (ioctl(sockfd, SIOCGIFADDR, &ifr) < 0) {
        perror("ioctl SIOCGIFADDR");
    } else {
        struct sockaddr_in *addr = (struct sockaddr_in *)&ifr.ifr_addr;
        g_config.if_addr = addr->sin_addr.s_addr;
        printf("[IGMP] 接口 IP: %s\n", inet_ntoa(addr->sin_addr));
    }

    /* 加入 224.0.0.1 组播组以接收所有 IGMP 报文 */
    struct ip_mreqn mreq;
    memset(&mreq, 0, sizeof(mreq));
    mreq.imr_multiaddr.s_addr = inet_addr(IGMP_ALL_HOSTS);
    mreq.imr_ifindex = g_config.if_index;
    if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        perror("setsockopt IP_ADD_MEMBERSHIP");
    }

    /* 加入 224.0.0.2 (所有路由器) */
    mreq.imr_multiaddr.s_addr = inet_addr(IGMP_ALL_ROUTERS);
    if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        perror("setsockopt IP_ADD_MEMBERSHIP (routers)");
    }

    g_config.sockfd = sockfd;
    printf("[IGMP] RAW Socket 创建成功 (fd=%d)\n", sockfd);
    
    return 0;
}

/**
 * 关闭 Socket
 */
void igmp_socket_close(void)
{
    if (g_config.sockfd > 0) {
        close(g_config.sockfd);
        g_config.sockfd = -1;
    }
}

/**
 * 发送 IGMP Query
 * @param group_addr 组地址 (0 表示 General Query)
 * @param general    1=General Query, 0=Group-Specific Query
 */
int igmp_send_query(uint32_t group_addr, int general)
{
    struct igmp_header igmp;
    struct sockaddr_in dest;
    int ret;

    /* 构造 IGMP Query */
    memset(&igmp, 0, sizeof(igmp));
    igmp.type = IGMP_MEMBERSHIP_QUERY;
    igmp.max_resp_time = IGMP_QUERY_RESPONSE_INTERVAL * 10;  /* 单位: 0.1秒 */
    igmp.group_addr = group_addr;  /* General Query 时为 0 */
    igmp.checksum = 0;
    igmp.checksum = igmp_checksum(&igmp, sizeof(igmp));

    /* 设置目的地址 */
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    
    if (general) {
        dest.sin_addr.s_addr = inet_addr(IGMP_ALL_HOSTS);  /* 224.0.0.1 */
    } else {
        dest.sin_addr.s_addr = group_addr;  /* 发送到具体组 */
    }

    /* 发送 */
    ret = sendto(g_config.sockfd, &igmp, sizeof(igmp), 0,
                 (struct sockaddr *)&dest, sizeof(dest));
    
    if (ret < 0) {
        perror("sendto");
        return -1;
    }

    if (g_config.verbose) {
        struct in_addr addr;
        addr.s_addr = group_addr;
        printf("[SEND] IGMP Query -> %s (group=%s)\n",
               inet_ntoa(dest.sin_addr),
               general ? "General" : inet_ntoa(addr));
    }

    return 0;
}

/**
 * 接收 IGMP 报文
 */
int igmp_recv_packet(void *buf, int buflen, struct sockaddr_in *from)
{
    socklen_t fromlen = sizeof(*from);
    int ret;

    ret = recvfrom(g_config.sockfd, buf, buflen, 0,
                   (struct sockaddr *)from, &fromlen);
    
    if (ret < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            perror("recvfrom");
        }
        return -1;
    }

    return ret;
}
