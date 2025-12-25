# 实验 1.3：RAW 套接口通信实验 - 运行指南

## 📁 文件结构

```
code/1.3/
├── raw_receiver.c   # RAW套接口接收端（监听ICMP包）
├── raw_sender.c     # RAW套接口发送端（发送ICMP Echo）
├── Makefile         # 编译脚本
└── README.md        # 本指南
```

---

## ⚠️ 重要提醒

**RAW 套接口需要 root 权限才能运行！**

---

## 🚀 运行步骤

### 1. 编译

```bash
cd 1.3
make
```

### 2. 运行测试

**方式一：使用接收端监听 ICMP 包**

```bash
# 终端1：启动接收端（需要sudo）
sudo ./raw_receiver

# 终端2：使用ping命令发送ICMP包
ping 127.0.0.1
```

**方式二：使用发送端发送 ICMP Echo Request**

```bash
# 直接运行发送端（类似ping）
sudo ./raw_sender
```

---

## 📸 截图要求

截图必须包含：

1. 程序运行结果（显示 ICMP 包信息）
2. 本机 MAC 地址（`ip link show | grep ether`）
3. 以上内容在同一张图内

---

## 📝 RAW 套接口特点

| 特性       | 说明                                      |
| ---------- | ----------------------------------------- |
| 权限       | 需要 root 权限                            |
| 协议       | 直接访问 IP 层，可处理 ICMP/TCP/UDP       |
| 用途       | 网络诊断工具（ping/traceroute）、协议分析 |
| 套接口类型 | `SOCK_RAW`                                |
