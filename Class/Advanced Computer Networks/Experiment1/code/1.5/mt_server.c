/**
 * Multi-threaded TCP Server - 实验1.5：多线程服务器
 * 
 * 功能：每个客户端连接由独立线程处理，实现并发服务
 * 参考：《UNIX网络编程》第26章 线程
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_PORT 8891
#define BUFFER_SIZE 1024
#define BACKLOG 10

// 客户端信息结构
typedef struct {
    int conn_fd;
    struct sockaddr_in addr;
    int client_id;
} client_info_t;

// 全局客户端计数器
int client_count = 0;
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;

// 客户端处理线程函数
void* handle_client(void* arg) {
    client_info_t* client = (client_info_t*)arg;
    char buffer[BUFFER_SIZE];
    ssize_t recv_len;
    
    printf("[Thread %d] Started for client %s:%d\n",
           client->client_id,
           inet_ntoa(client->addr.sin_addr),
           ntohs(client->addr.sin_port));

    // 发送欢迎消息
    char welcome[BUFFER_SIZE];
    snprintf(welcome, BUFFER_SIZE, "Welcome! You are client #%d\n", client->client_id);
    send(client->conn_fd, welcome, strlen(welcome), 0);

    // 循环处理客户端消息
    while (1) {
        memset(buffer, 0, BUFFER_SIZE);
        recv_len = recv(client->conn_fd, buffer, BUFFER_SIZE - 1, 0);
        
        if (recv_len <= 0) {
            printf("[Thread %d] Client disconnected.\n", client->client_id);
            break;
        }

        buffer[recv_len] = '\0';
        // 移除换行符
        buffer[strcspn(buffer, "\r\n")] = '\0';
        
        printf("[Thread %d] Received: %s\n", client->client_id, buffer);

        // 检查退出命令
        if (strcmp(buffer, "quit") == 0 || strcmp(buffer, "exit") == 0) {
            printf("[Thread %d] Client requested exit.\n", client->client_id);
            break;
        }

        // 回显消息
        char response[BUFFER_SIZE];
        snprintf(response, BUFFER_SIZE, "[Server Thread %d] Echo: %s\n", 
                 client->client_id, buffer);
        send(client->conn_fd, response, strlen(response), 0);
    }

    // 清理
    close(client->conn_fd);
    free(client);
    
    pthread_mutex_lock(&count_mutex);
    client_count--;
    printf("[Thread] Exiting. Active clients: %d\n", client_count);
    pthread_mutex_unlock(&count_mutex);
    
    return NULL;
}

int main() {
    int listen_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len;
    pthread_t tid;
    int opt = 1;

    printf("实验1.5：多线程TCP服务器\n");
    printf("========================\n\n");

    // 1. 创建监听套接口
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 2. 绑定地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);

    if (bind(listen_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }

    // 3. 开始监听
    if (listen(listen_fd, BACKLOG) < 0) {
        perror("listen failed");
        close(listen_fd);
        exit(EXIT_FAILURE);
    }

    printf("[Server] Listening on port %d\n", SERVER_PORT);
    printf("[Server] Waiting for connections...\n\n");

    // 4. 主循环：接受连接并创建线程
    while (1) {
        client_len = sizeof(client_addr);
        int conn_fd = accept(listen_fd, (struct sockaddr*)&client_addr, &client_len);
        
        if (conn_fd < 0) {
            perror("accept failed");
            continue;
        }

        // 分配客户端信息结构
        client_info_t* client = malloc(sizeof(client_info_t));
        if (client == NULL) {
            perror("malloc failed");
            close(conn_fd);
            continue;
        }

        pthread_mutex_lock(&count_mutex);
        client_count++;
        client->conn_fd = conn_fd;
        client->addr = client_addr;
        client->client_id = client_count;
        printf("[Server] New connection! Client #%d from %s:%d. Active: %d\n",
               client->client_id,
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port),
               client_count);
        pthread_mutex_unlock(&count_mutex);

        // 创建新线程处理客户端
        if (pthread_create(&tid, NULL, handle_client, client) != 0) {
            perror("pthread_create failed");
            close(conn_fd);
            free(client);
            continue;
        }

        // 分离线程，无需join
        pthread_detach(tid);
    }

    close(listen_fd);
    return 0;
}
