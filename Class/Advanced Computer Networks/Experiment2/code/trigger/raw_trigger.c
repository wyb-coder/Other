/**
 * RAW Socket 触发程序 - 实验二：触发网络设备驱动
 * 
 * 功能：使用RAW套接口向vnet0设备发送数据包，触发驱动的Tx函数
 * 需要root权限运行
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netpacket/packet.h>
#include <net/ethernet.h>
#include <arpa/inet.h>

#define DEVICE_NAME "vnet0"
#define PACKET_SIZE 64

int main(int argc, char *argv[])
{
    int sockfd;
    struct ifreq ifr;
    struct sockaddr_ll saddr;
    unsigned char packet[PACKET_SIZE];
    int i;
    
    printf("========================================\n");
    printf("  虚拟网卡驱动触发程序 - 实验二\n");
    printf("========================================\n\n");
    
    /* 1. 创建RAW套接口 */
    sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd < 0) {
        perror("socket创建失败 (需要root权限)");
        return -1;
    }
    printf("[INFO] RAW套接口创建成功\n");
    
    /* 2. 获取网卡接口索引 */
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, DEVICE_NAME, IFNAMSIZ - 1);
    
    if (ioctl(sockfd, SIOCGIFINDEX, &ifr) < 0) {
        perror("获取接口索引失败 (请确保vnet0已启动)");
        close(sockfd);
        return -1;
    }
    printf("[INFO] 设备 %s 接口索引: %d\n", DEVICE_NAME, ifr.ifr_ifindex);
    
    /* 3. 设置目标地址 */
    memset(&saddr, 0, sizeof(saddr));
    saddr.sll_family = AF_PACKET;
    saddr.sll_protocol = htons(ETH_P_ALL);
    saddr.sll_ifindex = ifr.ifr_ifindex;
    saddr.sll_halen = ETH_ALEN;
    
    /* 目标MAC地址 (vnet0的MAC) */
    saddr.sll_addr[0] = 0x00;
    saddr.sll_addr[1] = 0x40;
    saddr.sll_addr[2] = 0x63;
    saddr.sll_addr[3] = 0x80;
    saddr.sll_addr[4] = 0x00;
    saddr.sll_addr[5] = 0x00;
    
    /* 4. 构造测试数据包 */
    memset(packet, 0, PACKET_SIZE);
    
    /* 目标MAC */
    packet[0] = 0x00; packet[1] = 0x40; packet[2] = 0x63;
    packet[3] = 0x80; packet[4] = 0x00; packet[5] = 0x00;
    
    /* 源MAC */
    packet[6] = 0x00; packet[7] = 0x11; packet[8] = 0x22;
    packet[9] = 0x33; packet[10] = 0x44; packet[11] = 0x55;
    
    /* 以太网类型 (IP) */
    packet[12] = 0x08;
    packet[13] = 0x00;
    
    /* 填充测试数据 */
    for (i = 14; i < PACKET_SIZE; i++) {
        packet[i] = (unsigned char)(i - 14);
    }
    
    printf("\n[INFO] 准备发送测试数据包...\n");
    printf("[INFO] 数据包大小: %d 字节\n", PACKET_SIZE);
    printf("[INFO] 目标设备: %s\n\n", DEVICE_NAME);
    
    /* 5. 发送数据包 */
    printf("========================================\n");
    printf("发送 3 个测试数据包...\n");
    printf("========================================\n\n");
    
    for (i = 1; i <= 3; i++) {
        /* 修改数据包中的序号 */
        packet[14] = (unsigned char)i;
        
        if (sendto(sockfd, packet, PACKET_SIZE, 0,
                  (struct sockaddr*)&saddr, sizeof(saddr)) < 0) {
            perror("发送失败");
        } else {
            printf("[SEND] 第 %d 个数据包发送成功!\n", i);
        }
        
        usleep(500000);  /* 间隔500ms */
    }
    
    printf("\n========================================\n");
    printf("数据包发送完成!\n");
    printf("请使用 'dmesg | tail -50' 查看驱动打印\n");
    printf("========================================\n");
    
    close(sockfd);
    return 0;
}
