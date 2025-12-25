/**
 * UDP with Control Info - 实验1.4：高级套接口编程（控制信息）
 * 
 * 功能：演示如何使用recvmsg/sendmsg获取辅助数据（控制信息）
 * 包括：目的地址、接收接口、TTL等信息
 * 参考：《UNIX网络编程》第14章 高级I/O函数
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_PORT 8890
#define BUFFER_SIZE 1024

int main() {
    int sockfd;
    struct sockaddr_in server_addr, client_addr;
    char buffer[BUFFER_SIZE];
    char control[BUFFER_SIZE];
    struct msghdr msg;
    struct iovec iov[1];
    struct cmsghdr *cmsg;
    ssize_t recv_len;
    int opt = 1;

    printf("实验1.4：UDP控制信息演示\n");
    printf("=========================\n\n");

    // 1. 创建UDP套接口
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    printf("[Server] UDP socket created.\n");

    // 2. 启用接收控制信息选项
    // IP_PKTINFO: 获取目的地址和接收接口信息
    if (setsockopt(sockfd, IPPROTO_IP, IP_PKTINFO, &opt, sizeof(opt)) < 0) {
        perror("setsockopt IP_PKTINFO failed");
    } else {
        printf("[Server] IP_PKTINFO enabled.\n");
    }

    // IP_RECVTTL: 获取接收包的TTL值
    if (setsockopt(sockfd, IPPROTO_IP, IP_RECVTTL, &opt, sizeof(opt)) < 0) {
        perror("setsockopt IP_RECVTTL failed");
    } else {
        printf("[Server] IP_RECVTTL enabled.\n");
    }

    // 3. 绑定地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(SERVER_PORT);

    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(sockfd);
        exit(EXIT_FAILURE);
    }
    printf("[Server] Bindied to port %d.\n", SERVER_PORT);
    printf("[Server] Waiting for UDP packets...\n");
    printf("[Server] (Use: echo 'test' | nc -u 127.0.0.1 %d)\n\n", SERVER_PORT);

    // 4. 接收消息并获取控制信息
    int count = 0;
    while (count < 5) {
        memset(buffer, 0, BUFFER_SIZE);
        memset(control, 0, BUFFER_SIZE);
        memset(&msg, 0, sizeof(msg));
        memset(&client_addr, 0, sizeof(client_addr));

        // 设置消息头
        iov[0].iov_base = buffer;
        iov[0].iov_len = BUFFER_SIZE;
        
        msg.msg_name = &client_addr;
        msg.msg_namelen = sizeof(client_addr);
        msg.msg_iov = iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control;
        msg.msg_controllen = BUFFER_SIZE;

        // 使用 recvmsg 接收（可获取控制信息）
        recv_len = recvmsg(sockfd, &msg, 0);
        
        if (recv_len < 0) {
            perror("recvmsg failed");
            continue;
        }

        count++;
        printf("========== Packet %d ==========\n", count);
        printf("Data: %s", buffer);
        printf("From: %s:%d\n", 
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port));
        printf("Length: %zd bytes\n", recv_len);

        // 解析控制信息
        printf("\n[Control Information]\n");
        for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
            
            if (cmsg->cmsg_level == IPPROTO_IP && cmsg->cmsg_type == IP_PKTINFO) {
                struct in_pktinfo *pktinfo = (struct in_pktinfo*)CMSG_DATA(cmsg);
                printf("  Dest IP: %s\n", inet_ntoa(pktinfo->ipi_addr));
                printf("  Interface Index: %d\n", pktinfo->ipi_ifindex);
            }
            
            if (cmsg->cmsg_level == IPPROTO_IP && cmsg->cmsg_type == IP_TTL) {
                int ttl = *(int*)CMSG_DATA(cmsg);
                printf("  TTL: %d\n", ttl);
            }
        }
        printf("\n");
    }

    close(sockfd);
    printf("[Server] Socket closed.\n");
    
    return 0;
}
