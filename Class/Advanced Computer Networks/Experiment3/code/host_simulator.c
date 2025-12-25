/**
 * host_simulator.c - 主机模拟器
 * 实验三：IGMP协议软件（路由器端）开发
 * 
 * 功能：模拟主机加入/离开组播组，发送 IGMP Report/Leave
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define IGMP_V2_MEMBERSHIP_REPORT 0x16
#define IGMP_LEAVE_GROUP          0x17
#define IGMP_ALL_ROUTERS          "224.0.0.2"

/* IGMP 报文头 */
struct igmp_header {
    uint8_t  type;
    uint8_t  max_resp_time;
    uint16_t checksum;
    uint32_t group_addr;
} __attribute__((packed));

/* 计算校验和 */
uint16_t checksum(void *data, int len)
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

/* 发送 IGMP Report */
int send_report(int sockfd, const char *group)
{
    struct igmp_header igmp;
    struct sockaddr_in dest;

    memset(&igmp, 0, sizeof(igmp));
    igmp.type = IGMP_V2_MEMBERSHIP_REPORT;
    igmp.max_resp_time = 0;
    igmp.group_addr = inet_addr(group);
    igmp.checksum = 0;
    igmp.checksum = checksum(&igmp, sizeof(igmp));

    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_addr.s_addr = inet_addr("224.0.0.2");  /* Report 发送到所有路由器 */

    if (sendto(sockfd, &igmp, sizeof(igmp), 0,
               (struct sockaddr *)&dest, sizeof(dest)) < 0) {
        perror("sendto");
        return -1;
    }

    printf("[HOST] 发送 Membership Report -> %s\n", group);
    return 0;
}

/* 发送 IGMP Leave */
int send_leave(int sockfd, const char *group)
{
    struct igmp_header igmp;
    struct sockaddr_in dest;

    memset(&igmp, 0, sizeof(igmp));
    igmp.type = IGMP_LEAVE_GROUP;
    igmp.max_resp_time = 0;
    igmp.group_addr = inet_addr(group);
    igmp.checksum = 0;
    igmp.checksum = checksum(&igmp, sizeof(igmp));

    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_addr.s_addr = inet_addr(IGMP_ALL_ROUTERS);  /* Leave 发送到所有路由器 */

    if (sendto(sockfd, &igmp, sizeof(igmp), 0,
               (struct sockaddr *)&dest, sizeof(dest)) < 0) {
        perror("sendto");
        return -1;
    }

    printf("[HOST] 发送 Leave Group -> %s\n", group);
    return 0;
}

int main()
{
    int sockfd;
    unsigned char ttl = 1;
    char input[64];
    char group[32];

    printf("\n");
    printf("########################################\n");
    printf("#  IGMP 主机模拟器 - 实验三            #\n");
    printf("########################################\n");
    printf("\n");

    /* 检查 root 权限 */
    if (geteuid() != 0) {
        fprintf(stderr, "[ERROR] 需要 root 权限！\n");
        return 1;
    }

    /* 创建 RAW Socket */
    sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_IGMP);
    if (sockfd < 0) {
        perror("socket");
        return 1;
    }

    /* 设置 TTL */
    setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));

    printf("[HOST] Socket 创建成功\n");
    printf("\n");
    printf("命令说明:\n");
    printf("  join <组地址>   - 加入组播组 (发送 Report)\n");
    printf("  leave <组地址>  - 离开组播组 (发送 Leave)\n");
    printf("  quit            - 退出\n");
    printf("\n");
    printf("示例:\n");
    printf("  join 239.1.1.1\n");
    printf("  leave 239.1.1.1\n");
    printf("\n");

    while (1) {
        printf("host> ");
        fflush(stdout);

        if (fgets(input, sizeof(input), stdin) == NULL) {
            break;
        }

        input[strcspn(input, "\n")] = '\0';

        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            break;
        }

        if (strncmp(input, "join ", 5) == 0) {
            strncpy(group, input + 5, sizeof(group) - 1);
            send_report(sockfd, group);
        }
        else if (strncmp(input, "leave ", 6) == 0) {
            strncpy(group, input + 6, sizeof(group) - 1);
            send_leave(sockfd, group);
        }
        else if (strlen(input) > 0) {
            printf("[HOST] 未知命令: %s\n", input);
            printf("       使用 join/leave <组地址> 或 quit\n");
        }
    }

    close(sockfd);
    printf("[HOST] 已退出\n");
    return 0;
}
