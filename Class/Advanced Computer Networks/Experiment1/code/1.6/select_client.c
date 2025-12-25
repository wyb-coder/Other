/**
 * Select Client - 实验1.6：多路IO用法
 * 
 * 功能：使用select同时监听TCP/UDP套接口和标准输入
 * 同时连接TCP和UDP服务器
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_IP   "127.0.0.1"
#define TCP_PORT    8892
#define UDP_PORT    8893
#define BUFFER_SIZE 1024

int main() {
    int tcp_fd, udp_fd;
    struct sockaddr_in tcp_addr, udp_addr;
    char buffer[BUFFER_SIZE];
    fd_set read_fds;
    int max_fd;

    printf("实验1.6：多路IO客户端 (select)\n");
    printf("==============================\n\n");

    // ==================== 创建TCP套接口并连接 ====================
    tcp_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_fd < 0) {
        perror("TCP socket failed");
        exit(EXIT_FAILURE);
    }

    memset(&tcp_addr, 0, sizeof(tcp_addr));
    tcp_addr.sin_family = AF_INET;
    tcp_addr.sin_port = htons(TCP_PORT);
    inet_pton(AF_INET, SERVER_IP, &tcp_addr.sin_addr);

    if (connect(tcp_fd, (struct sockaddr*)&tcp_addr, sizeof(tcp_addr)) < 0) {
        perror("TCP connect failed");
        close(tcp_fd);
        exit(EXIT_FAILURE);
    }
    printf("[Client] TCP connected to %s:%d\n", SERVER_IP, TCP_PORT);

    // ==================== 创建UDP套接口 ====================
    udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd < 0) {
        perror("UDP socket failed");
        close(tcp_fd);
        exit(EXIT_FAILURE);
    }

    memset(&udp_addr, 0, sizeof(udp_addr));
    udp_addr.sin_family = AF_INET;
    udp_addr.sin_port = htons(UDP_PORT);
    inet_pton(AF_INET, SERVER_IP, &udp_addr.sin_addr);
    printf("[Client] UDP ready to send to %s:%d\n", SERVER_IP, UDP_PORT);

    printf("\n[Client] Commands:\n");
    printf("  t:<message>  - Send via TCP\n");
    printf("  u:<message>  - Send via UDP\n");
    printf("  quit         - Exit\n\n");

    max_fd = (tcp_fd > udp_fd) ? tcp_fd : udp_fd;
    if (STDIN_FILENO > max_fd) max_fd = STDIN_FILENO;

    // ==================== 主循环：使用select ====================
    while (1) {
        FD_ZERO(&read_fds);
        FD_SET(STDIN_FILENO, &read_fds);  // 监听标准输入
        FD_SET(tcp_fd, &read_fds);         // 监听TCP套接口
        FD_SET(udp_fd, &read_fds);         // 监听UDP套接口

        printf("> ");
        fflush(stdout);

        int ready = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if (ready < 0) {
            perror("select failed");
            break;
        }

        // 检查标准输入
        if (FD_ISSET(STDIN_FILENO, &read_fds)) {
            memset(buffer, 0, BUFFER_SIZE);
            if (fgets(buffer, BUFFER_SIZE, stdin) == NULL) break;
            buffer[strcspn(buffer, "\r\n")] = '\0';

            if (strcmp(buffer, "quit") == 0) {
                printf("[Client] Exiting...\n");
                break;
            }

            if (strncmp(buffer, "t:", 2) == 0) {
                // TCP发送
                send(tcp_fd, buffer + 2, strlen(buffer) - 2, 0);
                printf("[Client] Sent via TCP: %s\n", buffer + 2);
            } else if (strncmp(buffer, "u:", 2) == 0) {
                // UDP发送
                sendto(udp_fd, buffer + 2, strlen(buffer) - 2, 0,
                      (struct sockaddr*)&udp_addr, sizeof(udp_addr));
                printf("[Client] Sent via UDP: %s\n", buffer + 2);
            } else {
                printf("[Client] Use 't:msg' for TCP or 'u:msg' for UDP\n");
            }
        }

        // 检查TCP套接口（服务器回复）
        if (FD_ISSET(tcp_fd, &read_fds)) {
            memset(buffer, 0, BUFFER_SIZE);
            ssize_t n = recv(tcp_fd, buffer, BUFFER_SIZE - 1, 0);
            if (n <= 0) {
                printf("[Client] TCP server disconnected.\n");
                break;
            }
            printf("[TCP Response] %s", buffer);
        }

        // 检查UDP套接口（服务器回复）
        if (FD_ISSET(udp_fd, &read_fds)) {
            memset(buffer, 0, BUFFER_SIZE);
            ssize_t n = recv(udp_fd, buffer, BUFFER_SIZE - 1, 0);
            if (n > 0) {
                printf("[UDP Response] %s", buffer);
            }
        }
    }

    close(tcp_fd);
    close(udp_fd);
    printf("[Client] Goodbye!\n");
    
    return 0;
}
