/**
 * TCP Server - 实验1.2：TCP套接口通信实验
 * 
 * 功能：创建TCP服务器，接收客户端连接，接收消息并回显
 * 参考：《UNIX网络编程》
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_PORT 8889
#define BUFFER_SIZE 1024
#define BACKLOG 5

int main() {
    int listen_fd, conn_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len;
    char buffer[BUFFER_SIZE];
    ssize_t recv_len;

    // 1. 创建TCP套接口
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    printf("[Server] TCP socket created successfully.\n");

    // 设置地址重用，避免 "Address already in use" 错误
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 2. 初始化服务器地址结构
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);

    // 3. 绑定套接口到地址
    if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }
    printf("[Server] Bindied to port %d.\n", SERVER_PORT);

    // 4. 监听连接
    if (listen(listen_fd, BACKLOG) < 0) {
        perror("listen failed");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }
    printf("[Server] Listening for connections...\n");

    // 5. 接受客户端连接
    client_len = sizeof(client_addr);
    conn_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
    if (conn_fd < 0) {
        perror("accept failed");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }
    printf("[Server] Client connected from %s:%d\n",
           inet_ntoa(client_addr.sin_addr),
           ntohs(client_addr.sin_port));

    // 6. 循环接收和处理消息
    while (1) {
        memset(buffer, 0, BUFFER_SIZE);

        // 接收TCP数据
        recv_len = recv(conn_fd, buffer, BUFFER_SIZE - 1, 0);
        
        if (recv_len < 0) {
            perror("recv failed");
            break;
        } else if (recv_len == 0) {
            printf("[Server] Client disconnected.\n");
            break;
        }

        buffer[recv_len] = '\0';
        printf("[Server] Received: %s\n", buffer);

        // 检查退出命令
        if (strcmp(buffer, "quit") == 0 || strcmp(buffer, "exit") == 0) {
            printf("[Server] Received exit command. Shutting down...\n");
            break;
        }

        // 回显消息给客户端
        char response[BUFFER_SIZE];
        snprintf(response, BUFFER_SIZE, "[Echo] %s", buffer);
        
        if (send(conn_fd, response, strlen(response), 0) < 0) {
            perror("send failed");
        } else {
            printf("[Server] Sent response to client.\n");
        }
    }

    // 7. 关闭套接口
    close(conn_fd);
    close(listen_fd);
    printf("[Server] Sockets closed. Goodbye!\n");
    
    return 0;
}
