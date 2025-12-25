# =============================================================================
# IGMPv2 路由器 Makefile
# 实验三：IGMP协议软件（路由器端）开发
# =============================================================================

CC = gcc
CFLAGS = -g -Wall -O2

# 目标
TARGET = igmp_router
HOST_TARGET = host_simulator

# 源文件
SRCS = igmp_main.c igmp_socket.c igmp_proto.c igmp_group.c
OBJS = $(SRCS:.c=.o)

# 默认目标
all: $(TARGET) $(HOST_TARGET)

# 编译路由器程序
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "=========================================="
	@echo "[OK] 编译完成: $(TARGET)"
	@echo "=========================================="

# 编译主机模拟器
$(HOST_TARGET): host_simulator.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "[OK] 编译完成: $(HOST_TARGET)"

# 编译规则
%.o: %.c igmp.h
	$(CC) $(CFLAGS) -c $< -o $@

# 清理
clean:
	rm -f $(TARGET) $(HOST_TARGET) $(OBJS)
	@echo "[OK] 清理完成"

# 运行路由器 (需要 sudo)
run: $(TARGET)
	sudo ./$(TARGET) -i eth0 -v

# 运行主机模拟器 (需要 sudo)
run-host: $(HOST_TARGET)
	sudo ./$(HOST_TARGET)

# 帮助
help:
	@echo "使用说明:"
	@echo "  make          - 编译所有程序"
	@echo "  make clean    - 清理编译文件"
	@echo "  make run      - 运行 IGMP 路由器"
	@echo "  make run-host - 运行主机模拟器"
	@echo ""
	@echo "手动运行:"
	@echo "  虚拟机1 (路由器): sudo ./igmp_router -i eth0 -v"
	@echo "  虚拟机2 (主机):   sudo ./host_simulator"

.PHONY: all clean run run-host help
