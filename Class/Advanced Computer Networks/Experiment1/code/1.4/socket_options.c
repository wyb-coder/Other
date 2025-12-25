/**
 * Socket Options Demo - 实验1.4：高级套接口编程（选项和控制信息）
 * 
 * 功能：演示如何读取和设置各种套接口选项
 * 参考：《UNIX网络编程》第7章 套接口选项
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

// 打印分隔线
void print_separator(const char *title) {
    printf("\n========== %s ==========\n", title);
}

// 获取并打印整型选项值
void get_int_option(int sockfd, int level, int optname, const char *name) {
    int optval;
    socklen_t optlen = sizeof(optval);
    
    if (getsockopt(sockfd, level, optname, &optval, &optlen) < 0) {
        printf("  %-20s: [获取失败]\n", name);
    } else {
        printf("  %-20s: %d\n", name, optval);
    }
}

// 获取并打印缓冲区大小选项
void get_buffer_option(int sockfd, int level, int optname, const char *name) {
    int optval;
    socklen_t optlen = sizeof(optval);
    
    if (getsockopt(sockfd, level, optname, &optval, &optlen) < 0) {
        printf("  %-20s: [获取失败]\n", name);
    } else {
        printf("  %-20s: %d bytes\n", name, optval);
    }
}

// 设置整型选项值
int set_int_option(int sockfd, int level, int optname, int value, const char *name) {
    if (setsockopt(sockfd, level, optname, &value, sizeof(value)) < 0) {
        printf("  设置 %s 失败\n", name);
        return -1;
    }
    printf("  设置 %s = %d 成功\n", name, value);
    return 0;
}

int main() {
    int tcp_fd, udp_fd;
    
    printf("实验1.4：高级套接口编程（选项和控制信息）\n");
    printf("=============================================\n");

    // 创建TCP套接口
    tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd < 0) {
        perror("TCP socket creation failed");
        exit(EXIT_FAILURE);
    }

    // 创建UDP套接口
    udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd < 0) {
        perror("UDP socket creation failed");
        close(tcp_fd);
        exit(EXIT_FAILURE);
    }

    // ==================== 读取通用套接口选项 ====================
    print_separator("通用套接口选项 (SOL_SOCKET)");
    
    printf("\n[TCP 套接口]\n");
    get_int_option(tcp_fd, SOL_SOCKET, SO_REUSEADDR, "SO_REUSEADDR");
    get_int_option(tcp_fd, SOL_SOCKET, SO_KEEPALIVE, "SO_KEEPALIVE");
    get_buffer_option(tcp_fd, SOL_SOCKET, SO_SNDBUF, "SO_SNDBUF");
    get_buffer_option(tcp_fd, SOL_SOCKET, SO_RCVBUF, "SO_RCVBUF");
    get_int_option(tcp_fd, SOL_SOCKET, SO_TYPE, "SO_TYPE (1=STREAM)");
    
    printf("\n[UDP 套接口]\n");
    get_int_option(udp_fd, SOL_SOCKET, SO_REUSEADDR, "SO_REUSEADDR");
    get_int_option(udp_fd, SOL_SOCKET, SO_BROADCAST, "SO_BROADCAST");
    get_buffer_option(udp_fd, SOL_SOCKET, SO_SNDBUF, "SO_SNDBUF");
    get_buffer_option(udp_fd, SOL_SOCKET, SO_RCVBUF, "SO_RCVBUF");
    get_int_option(udp_fd, SOL_SOCKET, SO_TYPE, "SO_TYPE (2=DGRAM)");

    // ==================== 读取TCP特定选项 ====================
    print_separator("TCP 特定选项 (IPPROTO_TCP)");
    
    get_int_option(tcp_fd, IPPROTO_TCP, TCP_NODELAY, "TCP_NODELAY");
    get_int_option(tcp_fd, IPPROTO_TCP, TCP_MAXSEG, "TCP_MAXSEG");

    // ==================== 设置套接口选项 ====================
    print_separator("设置套接口选项");
    
    printf("\n[设置前 - TCP套接口]\n");
    get_int_option(tcp_fd, SOL_SOCKET, SO_REUSEADDR, "SO_REUSEADDR");
    get_int_option(tcp_fd, SOL_SOCKET, SO_KEEPALIVE, "SO_KEEPALIVE");
    get_int_option(tcp_fd, IPPROTO_TCP, TCP_NODELAY, "TCP_NODELAY");
    
    printf("\n[执行设置操作]\n");
    set_int_option(tcp_fd, SOL_SOCKET, SO_REUSEADDR, 1, "SO_REUSEADDR");
    set_int_option(tcp_fd, SOL_SOCKET, SO_KEEPALIVE, 1, "SO_KEEPALIVE");
    set_int_option(tcp_fd, IPPROTO_TCP, TCP_NODELAY, 1, "TCP_NODELAY");
    
    printf("\n[设置后 - TCP套接口]\n");
    get_int_option(tcp_fd, SOL_SOCKET, SO_REUSEADDR, "SO_REUSEADDR");
    get_int_option(tcp_fd, SOL_SOCKET, SO_KEEPALIVE, "SO_KEEPALIVE");
    get_int_option(tcp_fd, IPPROTO_TCP, TCP_NODELAY, "TCP_NODELAY");

    // ==================== 修改缓冲区大小 ====================
    print_separator("修改缓冲区大小");
    
    printf("\n[修改前]\n");
    get_buffer_option(tcp_fd, SOL_SOCKET, SO_SNDBUF, "SO_SNDBUF");
    get_buffer_option(tcp_fd, SOL_SOCKET, SO_RCVBUF, "SO_RCVBUF");
    
    printf("\n[执行修改]\n");
    set_int_option(tcp_fd, SOL_SOCKET, SO_SNDBUF, 65536, "SO_SNDBUF");
    set_int_option(tcp_fd, SOL_SOCKET, SO_RCVBUF, 65536, "SO_RCVBUF");
    
    printf("\n[修改后]\n");
    get_buffer_option(tcp_fd, SOL_SOCKET, SO_SNDBUF, "SO_SNDBUF");
    get_buffer_option(tcp_fd, SOL_SOCKET, SO_RCVBUF, "SO_RCVBUF");
    printf("  (注意：内核可能会将设置值加倍)\n");

    // ==================== 选项说明 ====================
    print_separator("常用选项说明");
    printf("\n");
    printf("  SO_REUSEADDR  : 允许重用本地地址和端口\n");
    printf("  SO_KEEPALIVE  : 保持连接活动，检测对端存活\n");
    printf("  SO_SNDBUF     : 发送缓冲区大小\n");
    printf("  SO_RCVBUF     : 接收缓冲区大小\n");
    printf("  SO_BROADCAST  : 允许发送广播数据报\n");
    printf("  TCP_NODELAY   : 禁用Nagle算法，减少小包延迟\n");
    printf("  TCP_MAXSEG    : TCP最大分段大小\n");

    // 关闭套接口
    close(tcp_fd);
    close(udp_fd);
    
    print_separator("实验完成");
    printf("\n");
    
    return 0;
}
