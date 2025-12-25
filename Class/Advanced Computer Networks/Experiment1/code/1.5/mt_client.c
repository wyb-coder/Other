/**
 * Multi-client Tester - 实验1.5：多线程服务器测试客户端
 * 
 * 功能：模拟多个客户端同时连接服务器
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_IP   "127.0.0.1"
#define SERVER_PORT 8891
#define BUFFER_SIZE 1024
#define NUM_CLIENTS 3

// 客户端线程函数
void* client_thread(void* arg) {
    int client_id = *(int*)arg;
    int sockfd;
    struct sockaddr_in server_addr;
    char send_buffer[BUFFER_SIZE];
    char recv_buffer[BUFFER_SIZE];
    ssize_t recv_len;

    // 创建套接口
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket failed");
        return NULL;
    }

    // 连接服务器
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &server_addr.sin_addr);

    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect failed");
        close(sockfd);
        return NULL;
    }

    printf("[Client %d] Connected to server.\n", client_id);

    // 接收欢迎消息
    recv_len = recv(sockfd, recv_buffer, BUFFER_SIZE - 1, 0);
    if (recv_len > 0) {
        recv_buffer[recv_len] = '\0';
        printf("[Client %d] Server: %s", client_id, recv_buffer);
    }

    // 发送几条测试消息
    for (int i = 1; i <= 3; i++) {
        snprintf(send_buffer, BUFFER_SIZE, "Hello from client %d, message %d", client_id, i);
        send(sockfd, send_buffer, strlen(send_buffer), 0);
        printf("[Client %d] Sent: %s\n", client_id, send_buffer);

        // 接收回复
        memset(recv_buffer, 0, BUFFER_SIZE);
        recv_len = recv(sockfd, recv_buffer, BUFFER_SIZE - 1, 0);
        if (recv_len > 0) {
            recv_buffer[recv_len] = '\0';
            printf("[Client %d] Received: %s", client_id, recv_buffer);
        }

        sleep(1);  // 间隔1秒
    }

    // 发送退出命令
    send(sockfd, "quit", 4, 0);
    printf("[Client %d] Sent quit command.\n", client_id);

    close(sockfd);
    printf("[Client %d] Disconnected.\n", client_id);
    
    return NULL;
}

int main() {
    pthread_t threads[NUM_CLIENTS];
    int client_ids[NUM_CLIENTS];

    printf("实验1.5：多客户端测试程序\n");
    printf("==========================\n");
    printf("启动 %d 个客户端同时连接服务器...\n\n", NUM_CLIENTS);

    // 创建多个客户端线程
    for (int i = 0; i < NUM_CLIENTS; i++) {
        client_ids[i] = i + 1;
        if (pthread_create(&threads[i], NULL, client_thread, &client_ids[i]) != 0) {
            perror("pthread_create failed");
        }
        usleep(100000);  // 错开100ms启动
    }

    // 等待所有客户端完成
    for (int i = 0; i < NUM_CLIENTS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("\n[Main] All clients finished.\n");
    return 0;
}
