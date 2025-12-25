/**
 * igmp.h - IGMPv2 路由器公共头文件
 * 实验三：IGMP协议软件（路由器端）开发
 */

#ifndef _IGMP_H_
#define _IGMP_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <net/if.h>

/*==============================================================================
 * 常量定义
 *============================================================================*/

/* IGMP 报文类型 */
#define IGMP_MEMBERSHIP_QUERY       0x11
#define IGMP_V1_MEMBERSHIP_REPORT   0x12
#define IGMP_V2_MEMBERSHIP_REPORT   0x16
#define IGMP_LEAVE_GROUP            0x17

/* 组播地址 */
#define IGMP_ALL_HOSTS      "224.0.0.1"   /* 所有主机 */
#define IGMP_ALL_ROUTERS    "224.0.0.2"   /* 所有路由器 */

/* 定时器参数 (秒) - RFC 2236 */
#define IGMP_QUERY_INTERVAL         125   /* Query 发送间隔 */
#define IGMP_QUERY_RESPONSE_INTERVAL 10   /* 最大响应时间 */
#define IGMP_GROUP_MEMBERSHIP_INTERVAL 260 /* 组成员超时时间 */
#define IGMP_LAST_MEMBER_QUERY_INTERVAL 1  /* Group-Specific Query 间隔 */
#define IGMP_LAST_MEMBER_QUERY_COUNT    2  /* Group-Specific Query 次数 */

/* 其他常量 */
#define MAX_GROUPS          100   /* 最大组数量 */
#define RECV_BUF_SIZE       1500  /* 接收缓冲区大小 */
#define IGMP_TTL            1     /* IGMP 报文 TTL */

/*==============================================================================
 * 数据结构
 *============================================================================*/

/* IGMP 报文头 (8 字节) */
struct igmp_header {
    uint8_t  type;           /* 报文类型 */
    uint8_t  max_resp_time;  /* 最大响应时间 (0.1秒) */
    uint16_t checksum;       /* 校验和 */
    uint32_t group_addr;     /* 组播组地址 */
} __attribute__((packed));

/* 组成员记录 */
struct group_entry {
    uint32_t group_addr;         /* 组播组地址 */
    char     group_str[16];      /* 组地址字符串 */
    time_t   expire_time;        /* 过期时间 */
    int      query_count;        /* 待发Query次数 */
    struct group_entry *next;    /* 链表指针 */
};

/* 全局配置 */
struct igmp_config {
    int      sockfd;             /* RAW Socket */
    char     ifname[32];         /* 接口名 */
    int      if_index;           /* 接口索引 */
    uint32_t if_addr;            /* 接口 IP 地址 */
    int      verbose;            /* 详细模式 */
    int      running;            /* 运行标志 */
};

/*==============================================================================
 * 全局变量声明
 *============================================================================*/

extern struct igmp_config g_config;
extern struct group_entry *g_group_list;

/*==============================================================================
 * 函数声明
 *============================================================================*/

/* igmp_socket.c */
int  igmp_socket_init(const char *ifname);
void igmp_socket_close(void);
int  igmp_send_query(uint32_t group_addr, int general);
int  igmp_recv_packet(void *buf, int buflen, struct sockaddr_in *from);

/* igmp_proto.c */
void igmp_process_packet(void *buf, int len, struct sockaddr_in *from);
uint16_t igmp_checksum(void *data, int len);

/* igmp_group.c */
void group_table_init(void);
void group_table_cleanup(void);
int  group_add(uint32_t addr);
int  group_remove(uint32_t addr);
struct group_entry* group_find(uint32_t addr);
void group_update_timer(uint32_t addr);
void group_check_expire(void);
void group_print_all(void);
int  group_count(void);

/* igmp_timer.c */
void timer_init(void);
void timer_check(void);

#endif /* _IGMP_H_ */
