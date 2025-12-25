/**
 * igmp_main.c - IGMPv2 路由器主程序
 * 实验三：IGMP协议软件（路由器端）开发
 */

#include "igmp.h"

/* 全局变量 */
struct igmp_config g_config;
struct group_entry *g_group_list = NULL;

/* 信号处理 */
static void signal_handler(int sig)
{
    printf("\n[IGMP] 收到信号 %d，正在退出...\n", sig);
    g_config.running = 0;
}

/* 打印使用说明 */
static void print_usage(const char *prog)
{
    printf("用法: %s [-i 接口名] [-v] [-h]\n", prog);
    printf("\n选项:\n");
    printf("  -i <interface>  指定网络接口 (默认: eth0)\n");
    printf("  -v              详细输出模式\n");
    printf("  -h              显示帮助\n");
    printf("\n示例:\n");
    printf("  sudo %s -i eth0 -v\n", prog);
}

/* 主函数 */
int main(int argc, char *argv[])
{
    int opt;
    char recv_buf[RECV_BUF_SIZE];
    struct sockaddr_in from_addr;
    int recv_len;
    time_t last_query_time = 0;
    time_t now;

    /* 初始化配置 */
    memset(&g_config, 0, sizeof(g_config));
    strcpy(g_config.ifname, "eth0");
    g_config.verbose = 0;
    g_config.running = 1;

    /* 解析命令行参数 */
    while ((opt = getopt(argc, argv, "i:vh")) != -1) {
        switch (opt) {
        case 'i':
            strncpy(g_config.ifname, optarg, sizeof(g_config.ifname) - 1);
            break;
        case 'v':
            g_config.verbose = 1;
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            return 0;
        }
    }

    /* 打印欢迎信息 */
    printf("\n");
    printf("########################################\n");
    printf("#  IGMPv2 路由器 - 实验三              #\n");
    printf("#  Advanced Computer Networks Lab      #\n");
    printf("########################################\n");
    printf("\n");

    /* 检查 root 权限 */
    if (geteuid() != 0) {
        fprintf(stderr, "[ERROR] 需要 root 权限运行！\n");
        fprintf(stderr, "请使用: sudo %s\n", argv[0]);
        return 1;
    }

    /* 初始化 Socket */
    printf("[IGMP] 初始化中...\n");
    printf("[IGMP] 网络接口: %s\n", g_config.ifname);
    
    if (igmp_socket_init(g_config.ifname) < 0) {
        fprintf(stderr, "[ERROR] Socket 初始化失败！\n");
        return 1;
    }

    /* 初始化组成员表 */
    group_table_init();

    /* 注册信号处理 */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("[IGMP] 路由器启动成功！\n");
    printf("[IGMP] 按 Ctrl+C 退出\n");
    printf("\n");

    /* 发送初始 General Query */
    printf("[IGMP] 发送初始 General Query...\n");
    igmp_send_query(0, 1);
    last_query_time = time(NULL);

    /* 主循环 */
    printf("\n[IGMP] 等待主机 IGMP Report...\n");
    printf("========================================\n\n");

    while (g_config.running) {
        /* 设置接收超时 */
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        setsockopt(g_config.sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        /* 接收报文 */
        recv_len = igmp_recv_packet(recv_buf, sizeof(recv_buf), &from_addr);
        
        if (recv_len > 0) {
            /* 处理报文 */
            igmp_process_packet(recv_buf, recv_len, &from_addr);
        }

        /* 检查定时器 */
        now = time(NULL);

        /* 周期性发送 General Query */
        if (now - last_query_time >= IGMP_QUERY_INTERVAL) {
            printf("[TIMER] 发送周期 General Query\n");
            igmp_send_query(0, 1);
            last_query_time = now;
        }

        /* 检查组成员超时 */
        group_check_expire();
    }

    /* 清理 */
    printf("\n[IGMP] 正在清理...\n");
    group_print_all();
    group_table_cleanup();
    igmp_socket_close();
    
    printf("[IGMP] 路由器已停止。\n");
    return 0;
}
