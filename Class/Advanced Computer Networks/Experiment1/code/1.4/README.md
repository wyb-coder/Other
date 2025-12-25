# 实验 1.4：高级套接口编程 - 运行指南

## 📁 文件结构

```
code/1.4/
├── socket_options.c     # 套接口选项读取和设置演示
├── udp_control_info.c   # UDP控制信息获取演示
├── Makefile
└── README.md
```

---

## 🚀 运行步骤

### 1. 编译

```bash
cd 1.4
make
```

### 2. 运行程序

**程序 1：套接口选项演示**

```bash
./socket_options
```

直接运行即可看到各种套接口选项的默认值和修改效果。

**程序 2：UDP 控制信息演示**

```bash
# 终端1：启动服务器
./udp_control_info

# 终端2：发送测试数据
echo "Hello" | nc -u 127.0.0.1 8890
```

---

## 📝 主要选项说明

| 选项         | 层级        | 说明               |
| ------------ | ----------- | ------------------ |
| SO_REUSEADDR | SOL_SOCKET  | 允许重用地址和端口 |
| SO_KEEPALIVE | SOL_SOCKET  | 保持连接活动检测   |
| SO_SNDBUF    | SOL_SOCKET  | 发送缓冲区大小     |
| SO_RCVBUF    | SOL_SOCKET  | 接收缓冲区大小     |
| TCP_NODELAY  | IPPROTO_TCP | 禁用 Nagle 算法    |
| IP_PKTINFO   | IPPROTO_IP  | 获取包目的地址信息 |
