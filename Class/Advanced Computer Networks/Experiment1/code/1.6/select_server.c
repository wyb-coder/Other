/**
 * Select Server - 实验1.6：多路IO用法
 * 
 * 功能：使用select同时监听TCP和UDP两个套接口
 * 参考：《UNIX网络编程》第6章 I/O复用
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define TCP_PORT 8892
#define UDP_PORT 8893
#define BUFFER_SIZE 1024
#define BACKLOG 5

int main() {
    int tcp_listen_fd, tcp_conn_fd = -1, udp_fd;
    struct sockaddr_in tcp_addr, udp_addr, client_addr;
    socklen_t client_len;
    char buffer[BUFFER_SIZE];
    fd_set read_fds, all_fds;
    int max_fd;
    int opt = 1;

    printf("实验1.6：多路IO服务器 (select)\n");
    printf("==============================\n\n");

    // ==================== 创建TCP监听套接口 ====================
    tcp_listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_listen_fd < 0) {
        perror("TCP socket failed");
        exit(EXIT_FAILURE);
    }
    setsockopt(tcp_listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&tcp_addr, 0, sizeof(tcp_addr));
    tcp_addr.sin_family = AF_INET;
    tcp_addr.sin_addr.s_addr = INADDR_ANY;
    tcp_addr.sin_port = htons(TCP_PORT);

    if (bind(tcp_listen_fd, (struct sockaddr*)&tcp_addr, sizeof(tcp_addr)) < 0) {
        perror("TCP bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(tcp_listen_fd, BACKLOG) < 0) {
        perror("TCP listen failed");
        exit(EXIT_FAILURE);
    }
    printf("[Server] TCP listening on port %d\n", TCP_PORT);

    // ==================== 创建UDP套接口 ====================
    udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd < 0) {
        perror("UDP socket failed");
        exit(EXIT_FAILURE);
    }

    memset(&udp_addr, 0, sizeof(udp_addr));
    udp_addr.sin_family = AF_INET;
    udp_addr.sin_addr.s_addr = INADDR_ANY;
    udp_addr.sin_port = htons(UDP_PORT);

    if (bind(udp_fd, (struct sockaddr*)&udp_addr, sizeof(udp_addr)) < 0) {
        perror("UDP bind failed");
        exit(EXIT_FAILURE);
    }
    printf("[Server] UDP listening on port %d\n", UDP_PORT);

    printf("\n[Server] Waiting for TCP/UDP connections...\n");
    printf("[Server] Test with:\n");
    printf("  TCP: nc 127.0.0.1 %d\n", TCP_PORT);
    printf("  UDP: echo 'hello' | nc -u 127.0.0.1 %d\n\n", UDP_PORT);

    // ==================== 初始化fd_set ====================
    FD_ZERO(&all_fds);
    FD_SET(tcp_listen_fd, &all_fds);
    FD_SET(udp_fd, &all_fds);
    max_fd = (tcp_listen_fd > udp_fd) ? tcp_listen_fd : udp_fd;

    // ==================== 主循环：使用select多路复用 ====================
    while (1) {
        read_fds = all_fds;  // select会修改fd_set，需要复制

        // select阻塞等待任一fd就绪
        int ready = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if (ready < 0) {
            perror("select failed");
            break;
        }

        // 检查TCP监听套接口（新连接）
        if (FD_ISSET(tcp_listen_fd, &read_fds)) {
            client_len = sizeof(client_addr);
            tcp_conn_fd = accept(tcp_listen_fd, (struct sockaddr*)&client_addr, &client_len);
            if (tcp_conn_fd >= 0) {
                printf("[TCP] New connection from %s:%d\n",
                       inet_ntoa(client_addr.sin_addr),
                       ntohs(client_addr.sin_port));
                
                FD_SET(tcp_conn_fd, &all_fds);
                if (tcp_conn_fd > max_fd) max_fd = tcp_conn_fd;
                
                send(tcp_conn_fd, "Welcome via TCP!\n", 17, 0);
            }
        }

        // 检查TCP已连接套接口（数据到达）
        if (tcp_conn_fd >= 0 && FD_ISSET(tcp_conn_fd, &read_fds)) {
            memset(buffer, 0, BUFFER_SIZE);
            ssize_t n = recv(tcp_conn_fd, buffer, BUFFER_SIZE - 1, 0);
            
            if (n <= 0) {
                printf("[TCP] Client disconnected.\n");
                FD_CLR(tcp_conn_fd, &all_fds);
                close(tcp_conn_fd);
                tcp_conn_fd = -1;
            } else {
                buffer[strcspn(buffer, "\r\n")] = '\0';
                printf("[TCP] Received: %s\n", buffer);
                
                char response[BUFFER_SIZE];
                snprintf(response, BUFFER_SIZE, "[TCP Echo] %s\n", buffer);
                send(tcp_conn_fd, response, strlen(response), 0);
            }
        }

        // 检查UDP套接口
        if (FD_ISSET(udp_fd, &read_fds)) {
            client_len = sizeof(client_addr);
            memset(buffer, 0, BUFFER_SIZE);
            ssize_t n = recvfrom(udp_fd, buffer, BUFFER_SIZE - 1, 0,
                                (struct sockaddr*)&client_addr, &client_len);
            
            if (n > 0) {
                buffer[strcspn(buffer, "\r\n")] = '\0';
                printf("[UDP] Received from %s:%d: %s\n",
                       inet_ntoa(client_addr.sin_addr),
                       ntohs(client_addr.sin_port),
                       buffer);
                
                char response[BUFFER_SIZE];
                snprintf(response, BUFFER_SIZE, "[UDP Echo] %s\n", buffer);
                sendto(udp_fd, response, strlen(response), 0,
                      (struct sockaddr*)&client_addr, client_len);
            }
        }
    }

    close(tcp_listen_fd);
    close(udp_fd);
    if (tcp_conn_fd >= 0) close(tcp_conn_fd);
    
    return 0;
}
