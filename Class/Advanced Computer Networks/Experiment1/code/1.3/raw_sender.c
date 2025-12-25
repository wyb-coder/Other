/**
 * RAW Socket Sender (Ping) - 实验1.3：RAW套接口通信实验
 * 
 * 功能：创建RAW套接口发送ICMP Echo Request（类似ping）
 * 注意：需要root权限运行
 * 参考：《UNIX网络编程》
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <arpa/inet.h>
#include <sys/time.h>

#define PACKET_SIZE 64
#define TARGET_IP   "127.0.0.1"

// 计算校验和
unsigned short checksum(void *data, int len) {
    unsigned short *buf = data;
    unsigned int sum = 0;
    unsigned short result;

    for (sum = 0; len > 1; len -= 2)
        sum += *buf++;
    if (len == 1)
        sum += *(unsigned char*)buf;
    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    result = ~sum;
    return result;
}

int main() {
    int sockfd;
    struct sockaddr_in dest_addr;
    char send_buffer[PACKET_SIZE];
    char recv_buffer[PACKET_SIZE * 2];
    struct icmphdr *icmp;
    int seq = 0;
    ssize_t send_len, recv_len;
    socklen_t addr_len;
    struct timeval tv;

    // 1. 创建RAW套接口
    sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sockfd < 0) {
        perror("socket creation failed (need root privilege)");
        exit(EXIT_FAILURE);
    }
    printf("[Sender] RAW socket created successfully.\n");

    // 设置接收超时
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // 2. 初始化目标地址
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.sin_family = AF_INET;
    inet_pton(AF_INET, TARGET_IP, &dest_addr.sin_addr);

    printf("[Sender] Sending ICMP Echo Requests to %s\n", TARGET_IP);
    printf("============================================\n\n");

    // 3. 发送5个ICMP Echo Request
    for (int i = 0; i < 5; i++) {
        memset(send_buffer, 0, PACKET_SIZE);
        
        // 构造ICMP头
        icmp = (struct icmphdr*)send_buffer;
        icmp->type = ICMP_ECHO;        // Echo Request
        icmp->code = 0;
        icmp->un.echo.id = htons(getpid() & 0xFFFF);
        icmp->un.echo.sequence = htons(++seq);
        icmp->checksum = 0;
        icmp->checksum = checksum(send_buffer, PACKET_SIZE);

        // 发送ICMP包
        send_len = sendto(sockfd, send_buffer, PACKET_SIZE, 0,
                         (struct sockaddr*)&dest_addr, sizeof(dest_addr));
        
        if (send_len < 0) {
            perror("sendto failed");
            continue;
        }
        printf("[Sender] Sent ICMP Echo Request, seq=%d\n", seq);

        // 接收ICMP Echo Reply
        addr_len = sizeof(dest_addr);
        recv_len = recvfrom(sockfd, recv_buffer, sizeof(recv_buffer), 0,
                           (struct sockaddr*)&dest_addr, &addr_len);
        
        if (recv_len < 0) {
            printf("[Sender] No reply (timeout)\n");
        } else {
            // 跳过IP头，解析ICMP回复
            struct iphdr *ip = (struct iphdr*)recv_buffer;
            struct icmphdr *reply = (struct icmphdr*)(recv_buffer + ip->ihl * 4);
            
            if (reply->type == ICMP_ECHOREPLY) {
                printf("[Sender] Received Echo Reply, seq=%d, ttl=%d\n",
                       ntohs(reply->un.echo.sequence), ip->ttl);
            }
        }
        printf("\n");
        sleep(1);  // 每秒发送一个
    }

    // 4. 关闭套接口
    close(sockfd);
    printf("[Sender] Socket closed. Goodbye!\n");
    
    return 0;
}
