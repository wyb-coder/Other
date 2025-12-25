/**
 * UDP Server - 实验1：UDP套接口通信实验
 * 
 * 功能：创建UDP服务器，接收客户端消息并回显
 * 参考：《UNIX网络编程》
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_PORT 8888
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len;
    char buffer[BUFFER_SIZE];
    ssize_t recv_len;

    // 1. 创建UDP套接口
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    printf("[Server] UDP socket created successfully.\n");

    // 2. 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;           // IPv4
    server_addr.sin_addr.s_addr = INADDR_ANY;   // 接受任意地址
    server_addr.sin_port = htons(SERVER_PORT);  // 端口号

    // 3. 绑定套接口到地址
    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    printf("[Server] Bindied to port %d, waiting for messages...\n", SERVER_PORT);

    // 4. 循环接收和处理消息
    while (1) {
        client_len = sizeof(client_addr);
        memset(buffer, 0, BUFFER_SIZE);

        // 接收UDP数据包
        recv_len = recvfrom(sockfd, buffer, BUFFER_SIZE - 1, 0,
                           (struct sockaddr*)&client_addr, &client_len);
        
        if (recv_len < 0) {
            perror("recvfrom failed");
            continue;
        }

        buffer[recv_len] = '\0';
        printf("[Server] Received from %s:%d: %s\n",
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port),
               buffer);

        // 检查退出命令
        if (strcmp(buffer, "quit") == 0 || strcmp(buffer, "exit") == 0) {
            printf("[Server] Received exit command. Shutting down...\n");
            break;
        }

        // 回显消息给客户端
        char response[BUFFER_SIZE];
        snprintf(response, BUFFER_SIZE, "[Echo] %s", buffer);
        
        if (sendto(sockfd, response, strlen(response), 0,
                  (struct sockaddr*)&client_addr, client_len) < 0) {
            perror("sendto failed");
        } else {
            printf("[Server] Sent response to client.\n");
        }
    }

    // 5. 关闭套接口
    close(sockfd);
    printf("[Server] Socket closed. Goodbye!\n");
    
    return 0;
}
