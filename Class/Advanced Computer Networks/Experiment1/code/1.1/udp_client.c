/**
 * UDP Client - 实验1：UDP套接口通信实验
 * 
 * 功能：创建UDP客户端，发送消息到服务器并接收回复
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
#define SERVER_PORT 8888
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr;
    socklen_t server_len;
    char send_buffer[BUFFER_SIZE];
    char recv_buffer[BUFFER_SIZE];
    ssize_t recv_len;

    // 1. 创建UDP套接口
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    printf("[Client] UDP socket created successfully.\n");

    // 2. 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    
    if (inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr) <= 0) {
        perror("Invalid address");
        close(sockfd);
        exit(EXIT_FAILURE);
    }

    printf("[Client] Ready to send messages to %s:%d\n", SERVER_IP, SERVER_PORT);
    printf("[Client] Type 'quit' or 'exit' to stop.\n");
    printf("----------------------------------------\n");

    // 3. 循环发送消息
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

        // 发送UDP数据包到服务器
        server_len = sizeof(server_addr);
        if (sendto(sockfd, send_buffer, strlen(send_buffer), 0,
                  (struct sockaddr*)&server_addr, server_len) < 0) {
            perror("sendto failed");
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
        recv_len = recvfrom(sockfd, recv_buffer, BUFFER_SIZE - 1, 0,
                           (struct sockaddr*)&server_addr, &server_len);
        
        if (recv_len < 0) {
            perror("recvfrom failed");
        } else {
            recv_buffer[recv_len] = '\0';
            printf("[Client] Server response: %s\n", recv_buffer);
        }
        printf("----------------------------------------\n");
    }

    // 4. 关闭套接口
    close(sockfd);
    printf("[Client] Socket closed. Goodbye!\n");
    
    return 0;
}
