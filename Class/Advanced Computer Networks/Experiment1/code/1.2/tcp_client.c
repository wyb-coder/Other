/**
 * TCP Client - 实验1.2：TCP套接口通信实验
 * 
 * 功能：创建TCP客户端，连接服务器，发送消息并接收回复
 * 参考：《UNIX网络编程》
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_IP   "127.0.0.1"
#define SERVER_PORT 8889
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    char send_buffer[BUFFER_SIZE];
    char recv_buffer[BUFFER_SIZE];
    ssize_t recv_len;

    // 1. 创建TCP套接口
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    printf("[Client] TCP socket created successfully.\n");

    // 2. 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) <= 0) {
        perror("Invalid address");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    // 3. 连接到服务器
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connection failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    printf("[Client] Connected to server %s:%d\n", SERVER_IP, SERVER_PORT);
    printf("[Client] Type 'quit' or 'exit' to stop.\n");
    printf("----------------------------------------\n");

    // 4. 循环发送消息
    while (1) {
        printf("Enter message: ");
        fflush(stdout);
        
        // 读取用户输入
        if (fgets(send_buffer, BUFFER_SIZE, stdin) == NULL) {
            break;
        }
        
        // 移除换行符
        send_buffer[strcspn(send_buffer, "\n")] = '\0';
        
        if (strlen(send_buffer) == 0) {
            continue;
        }

        // 发送TCP数据到服务器
        if (send(sockfd, send_buffer, strlen(send_buffer), 0) < 0) {
            perror("send failed");
            continue;
        }
        printf("[Client] Message sent to server.\n");

        // 检查退出命令
        if (strcmp(send_buffer, "quit") == 0 || strcmp(send_buffer, "exit") == 0) {
            printf("[Client] Exiting...\n");
            break;
        }

        // 接收服务器回复
        memset(recv_buffer, 0, BUFFER_SIZE);
        recv_len = recv(sockfd, recv_buffer, BUFFER_SIZE - 1, 0);
        
        if (recv_len < 0) {
            perror("recv failed");
        } else if (recv_len == 0) {
            printf("[Client] Server closed connection.\n");
            break;
        } else {
            recv_buffer[recv_len] = '\0';
            printf("[Client] Server response: %s\n", recv_buffer);
        }
        printf("----------------------------------------\n");
    }

    // 5. 关闭套接口
    close(sockfd);
    printf("[Client] Socket closed. Goodbye!\n");
    
    return 0;
}
