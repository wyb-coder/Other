/**
 * 虚拟网络设备驱动程序 - 实验二 (兼容 Linux 6.x 内核)
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/errno.h>
#include <linux/types.h>
#include <linux/init.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/skbuff.h>
#include <linux/version.h>

#define DRV_NAME        "vnetdev"
#define DRV_VERSION     "2.0"

MODULE_AUTHOR("Advanced Network Lab");
MODULE_DESCRIPTION("Virtual Network Device Driver");
MODULE_LICENSE("GPL");

static struct net_device *g_vnetdev = NULL;
static unsigned char g_dev_mac[ETH_ALEN] = {0x00, 0x40, 0x63, 0x80, 0x00, 0x00};

struct vnetdev_priv {
    struct net_device_stats stats;
    spinlock_t lock;
};

/* 打开网卡 - ifconfig vnet0 up */
static int Vnetdev_Open(struct net_device *dev)
{
    printk(KERN_EMERG "========================================\n");
    printk(KERN_EMERG "[VNETDEV] Vnetdev_Open() 被调用!\n");
    printk(KERN_EMERG "[VNETDEV] 设备: %s 正在启动...\n", dev->name);
    printk(KERN_EMERG "========================================\n");
    
    netif_start_queue(dev);
    return 0;
}

/* 关闭网卡 - ifconfig vnet0 down */
static int Vnetdev_Release(struct net_device *dev)
{
    printk(KERN_EMERG "========================================\n");
    printk(KERN_EMERG "[VNETDEV] Vnetdev_Release() 被调用!\n");
    printk(KERN_EMERG "[VNETDEV] 设备: %s 正在关闭...\n", dev->name);
    printk(KERN_EMERG "========================================\n");
    
    netif_stop_queue(dev);
    return 0;
}

/* 发送数据包 */
static netdev_tx_t Vnetdev_Tx(struct sk_buff *skb, struct net_device *dev)
{
    struct vnetdev_priv *priv = netdev_priv(dev);
    int i, len;
    
    printk(KERN_EMERG "========================================\n");
    printk(KERN_EMERG "[VNETDEV] Vnetdev_Tx() 被调用!\n");
    printk(KERN_EMERG "[VNETDEV] 数据包长度: %d 字节\n", skb->len);
    
    len = skb->len > 32 ? 32 : skb->len;
    printk(KERN_EMERG "[VNETDEV] 前%d字节: ", len);
    for (i = 0; i < len; i++)
        printk(KERN_CONT "%02x ", skb->data[i]);
    printk(KERN_CONT "\n");
    printk(KERN_EMERG "========================================\n");
    
    priv->stats.tx_packets++;
    priv->stats.tx_bytes += skb->len;
    dev_kfree_skb(skb);
    
    return NETDEV_TX_OK;
}

/* 获取统计 - ifconfig vnet0 */
static void Vnetdev_Stats(struct net_device *dev,
                          struct rtnl_link_stats64 *stats)
{
    struct vnetdev_priv *priv = netdev_priv(dev);
    
    printk(KERN_EMERG "[VNETDEV] Vnetdev_Stats() 被调用!\n");
    
    stats->tx_packets = priv->stats.tx_packets;
    stats->tx_bytes = priv->stats.tx_bytes;
    stats->rx_packets = priv->stats.rx_packets;
    stats->rx_bytes = priv->stats.rx_bytes;
}

/* 修改MTU - ifconfig vnet0 mtu 1222 */
static int Vnetdev_ChangeMtu(struct net_device *dev, int new_mtu)
{
    printk(KERN_EMERG "========================================\n");
    printk(KERN_EMERG "[VNETDEV] Vnetdev_ChangeMtu() 被调用!\n");
    printk(KERN_EMERG "[VNETDEV] MTU: %d -> %d\n", dev->mtu, new_mtu);
    printk(KERN_EMERG "========================================\n");
    
    dev->mtu = new_mtu;
    return 0;
}

/* 发送超时 (兼容新旧内核) */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,6,0)
static void Vnetdev_Timeout(struct net_device *dev, unsigned int txqueue)
#else
static void Vnetdev_Timeout(struct net_device *dev)
#endif
{
    printk(KERN_EMERG "[VNETDEV] Vnetdev_Timeout() 发送超时!\n");
}

/* 网络设备操作函数集 */
static const struct net_device_ops vnetdev_ops = {
    .ndo_open       = Vnetdev_Open,
    .ndo_stop       = Vnetdev_Release,
    .ndo_start_xmit = Vnetdev_Tx,
    .ndo_get_stats64 = Vnetdev_Stats,
    .ndo_change_mtu = Vnetdev_ChangeMtu,
    .ndo_tx_timeout = Vnetdev_Timeout,
};

/* 设备初始化 */
static void Vnetdev_Setup(struct net_device *dev)
{
    struct vnetdev_priv *priv;
    
    printk(KERN_EMERG "[VNETDEV] 初始化网络设备...\n");
    
    ether_setup(dev);
    dev->netdev_ops = &vnetdev_ops;
    dev->flags |= IFF_NOARP;
    
    /* 设置MAC地址 */
    eth_hw_addr_set(dev, g_dev_mac);
    
    priv = netdev_priv(dev);
    memset(priv, 0, sizeof(struct vnetdev_priv));
    spin_lock_init(&priv->lock);
    
    printk(KERN_EMERG "[VNETDEV] MAC: %02x:%02x:%02x:%02x:%02x:%02x\n",
           dev->dev_addr[0], dev->dev_addr[1], dev->dev_addr[2],
           dev->dev_addr[3], dev->dev_addr[4], dev->dev_addr[5]);
}

/* 模块加载 - insmod netdrv.ko */
static int __init Vnetdev_InitModule(void)
{
    int ret;
    
    printk(KERN_EMERG "\n########################################\n");
    printk(KERN_EMERG "#  虚拟网络设备驱动 - 实验二           #\n");
    printk(KERN_EMERG "########################################\n");
    
    g_vnetdev = alloc_netdev(sizeof(struct vnetdev_priv), "vnet%d",
                             NET_NAME_UNKNOWN, Vnetdev_Setup);
    if (!g_vnetdev)
        return -ENOMEM;
    
    ret = register_netdev(g_vnetdev);
    if (ret) {
        free_netdev(g_vnetdev);
        return ret;
    }
    
    printk(KERN_EMERG "[VNETDEV] 设备 '%s' 注册成功!\n", g_vnetdev->name);
    printk(KERN_EMERG "########################################\n\n");
    return 0;
}

/* 模块卸载 - rmmod netdrv */
static void __exit Vnetdev_Cleanup(void)
{
    printk(KERN_EMERG "\n[VNETDEV] 卸载驱动...\n");
    if (g_vnetdev) {
        unregister_netdev(g_vnetdev);
        free_netdev(g_vnetdev);
    }
    printk(KERN_EMERG "[VNETDEV] 驱动已卸载!\n\n");
}

module_init(Vnetdev_InitModule);
module_exit(Vnetdev_Cleanup);
