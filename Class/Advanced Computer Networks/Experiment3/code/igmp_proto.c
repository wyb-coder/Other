/**
 * igmp_proto.c - IGMP 协议处理模块
 * 实验三：IGMP协议软件（路由器端）开发
 */

#include "igmp.h"

/**
 * 计算校验和
 */
uint16_t igmp_checksum(void *data, int len)
{
    uint16_t *buf = (uint16_t *)data;
    uint32_t sum = 0;

    while (len > 1) {
        sum += *buf++;
        len -= 2;
    }

    if (len == 1) {
        sum += *(uint8_t *)buf;
    }

    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);

    return (uint16_t)(~sum);
}

/**
 * 获取 IGMP 类型名称
 */
static const char* igmp_type_name(uint8_t type)
{
    switch (type) {
    case IGMP_MEMBERSHIP_QUERY:
        return "Membership Query";
    case IGMP_V1_MEMBERSHIP_REPORT:
        return "V1 Membership Report";
    case IGMP_V2_MEMBERSHIP_REPORT:
        return "V2 Membership Report";
    case IGMP_LEAVE_GROUP:
        return "Leave Group";
    default:
        return "Unknown";
    }
}

/**
 * 处理 Membership Report
 */
static void igmp_process_report(struct igmp_header *igmp, struct sockaddr_in *from)
{
    struct in_addr group;
    group.s_addr = igmp->group_addr;

    printf("[RECV] ★ Membership Report from %s\n", inet_ntoa(from->sin_addr));
    printf("       组播组地址: %s\n", inet_ntoa(group));

    /* 检查是否为有效组播地址 (224.0.0.0 - 239.255.255.255) */
    uint32_t addr = ntohl(igmp->group_addr);
    if ((addr & 0xF0000000) != 0xE0000000) {
        printf("       [警告] 无效的组播地址，忽略\n");
        return;
    }

    /* 添加或更新组成员记录 */
    if (group_find(igmp->group_addr) == NULL) {
        group_add(igmp->group_addr);
        printf("       [新组] 已添加到组成员表\n");
    } else {
        group_update_timer(igmp->group_addr);
        printf("       [更新] 已更新超时时间\n");
    }

    /* 打印当前组成员表 */
    printf("\n");
    group_print_all();
    printf("\n");
}

/**
 * 处理 Leave Group
 */
static void igmp_process_leave(struct igmp_header *igmp, struct sockaddr_in *from)
{
    struct in_addr group;
    group.s_addr = igmp->group_addr;

    printf("[RECV] Leave Group from %s\n", inet_ntoa(from->sin_addr));
    printf("       组播组地址: %s\n", inet_ntoa(group));

    /* 发送 Group-Specific Query 确认 */
    printf("       发送 Group-Specific Query 确认...\n");
    
    for (int i = 0; i < IGMP_LAST_MEMBER_QUERY_COUNT; i++) {
        igmp_send_query(igmp->group_addr, 0);
        if (i < IGMP_LAST_MEMBER_QUERY_COUNT - 1) {
            sleep(IGMP_LAST_MEMBER_QUERY_INTERVAL);
        }
    }
}

/**
 * 处理 Membership Query (来自其他路由器)
 */
static void igmp_process_query(struct igmp_header *igmp, struct sockaddr_in *from)
{
    /* 如果是自己发送的 Query，忽略 */
    if (from->sin_addr.s_addr == g_config.if_addr) {
        return;
    }

    if (g_config.verbose) {
        struct in_addr group;
        group.s_addr = igmp->group_addr;
        printf("[RECV] Query from %s (group=%s)\n",
               inet_ntoa(from->sin_addr),
               igmp->group_addr ? inet_ntoa(group) : "General");
    }
}

/**
 * 处理接收到的 IGMP 报文
 */
void igmp_process_packet(void *buf, int len, struct sockaddr_in *from)
{
    struct iphdr *ip = (struct iphdr *)buf;
    struct igmp_header *igmp;
    int ip_hlen;

    /* 检查长度 */
    if (len < (int)sizeof(struct iphdr)) {
        return;
    }

    /* 跳过 IP 头 */
    ip_hlen = ip->ihl * 4;
    if (len < ip_hlen + (int)sizeof(struct igmp_header)) {
        return;
    }

    /* 验证协议 */
    if (ip->protocol != IPPROTO_IGMP) {
        return;
    }

    igmp = (struct igmp_header *)((char *)buf + ip_hlen);

    /* 验证校验和 */
    uint16_t orig_cksum = igmp->checksum;
    igmp->checksum = 0;
    uint16_t calc_cksum = igmp_checksum(igmp, sizeof(struct igmp_header));
    igmp->checksum = orig_cksum;

    if (orig_cksum != calc_cksum) {
        if (g_config.verbose) {
            printf("[WARN] 校验和错误: 收到 0x%04x, 计算 0x%04x\n",
                   ntohs(orig_cksum), ntohs(calc_cksum));
        }
        /* 继续处理，某些实现可能有问题 */
    }

    /* 根据类型处理 */
    if (g_config.verbose) {
        printf("[IGMP] 类型: %s (0x%02x)\n", igmp_type_name(igmp->type), igmp->type);
    }

    switch (igmp->type) {
    case IGMP_MEMBERSHIP_QUERY:
        igmp_process_query(igmp, from);
        break;

    case IGMP_V1_MEMBERSHIP_REPORT:
    case IGMP_V2_MEMBERSHIP_REPORT:
        igmp_process_report(igmp, from);
        break;

    case IGMP_LEAVE_GROUP:
        igmp_process_leave(igmp, from);
        break;

    default:
        if (g_config.verbose) {
            printf("[IGMP] 未知类型: 0x%02x\n", igmp->type);
        }
        break;
    }
}
