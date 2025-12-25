/**
 * RAW Socket Receiver - 实验1.3：RAW套接口通信实验
 * 
 * 功能：创建RAW套接口接收ICMP数据包
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

#define BUFFER_SIZE 65535

// 打印IP头信息
void print_ip_header(struct iphdr *ip) {
    struct in_addr src, dst;
    src.s_addr = ip->saddr;
    dst.s_addr = ip->daddr;
    
    printf("=== IP Header ===\n");
    printf("  Version: %d\n", ip->version);
    printf("  Header Length: %d bytes\n", ip->ihl * 4);
    printf("  TTL: %d\n", ip->ttl);
    printf("  Protocol: %d (ICMP=1, TCP=6, UDP=17)\n", ip->protocol);
    printf("  Source IP: %s\n", inet_ntoa(src));
    printf("  Dest IP: %s\n", inet_ntoa(dst));
}

// 打印ICMP头信息
void print_icmp_header(struct icmphdr *icmp) {
    printf("=== ICMP Header ===\n");
    printf("  Type: %d ", icmp->type);
    if (icmp->type == ICMP_ECHO) printf("(Echo Request)\n");
    else if (icmp->type == ICMP_ECHOREPLY) printf("(Echo Reply)\n");
    else printf("(Other)\n");
    printf("  Code: %d\n", icmp->code);
    printf("  ID: %d\n", ntohs(icmp->un.echo.id));
    printf("  Sequence: %d\n", ntohs(icmp->un.echo.sequence));
}

int main() {
    int sockfd;
    char buffer[BUFFER_SIZE];
    struct sockaddr_in src_addr;
    socklen_t addr_len;
    ssize_t recv_len;
    int count = 0;

    // 1. 创建RAW套接口（接收ICMP协议）
    sockfd = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP);
    if (sockfd < 0) {
        perror("socket creation failed (need root privilege)");
        exit(EXIT_FAILURE);
    }
    printf("[Receiver] RAW socket created successfully.\n");
    printf("[Receiver] Waiting for ICMP packets...\n");
    printf("[Receiver] (Use 'ping 127.0.0.1' in another terminal to test)\n");
    printf("============================================\n\n");

    // 2. 循环接收ICMP数据包
    while (count < 10) {  // 接收10个包后退出
        memset(buffer, 0, BUFFER_SIZE);
        addr_len = sizeof(src_addr);

        // 接收RAW数据包（包含IP头）
        recv_len = recvfrom(sockfd, buffer, BUFFER_SIZE, 0,
                           (struct sockaddr*)&src_addr, &addr_len);
        
        if (recv_len < 0) {
            perror("recvfrom failed");
            continue;
        }

        printf("[Packet %d] Received %zd bytes\n", ++count, recv_len);

        // 解析IP头
        struct iphdr *ip = (struct iphdr*)buffer;
        print_ip_header(ip);

        // 解析ICMP头（跳过IP头）
        struct icmphdr *icmp = (struct icmphdr*)(buffer + ip->ihl * 4);
        print_icmp_header(icmp);

        printf("\n");
    }

    // 3. 关闭套接口
    close(sockfd);
    printf("[Receiver] Socket closed. Received %d packets.\n", count);
    
    return 0;
}
