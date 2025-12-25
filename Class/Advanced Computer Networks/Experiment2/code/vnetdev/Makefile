# =============================================================================
# Linux 网络设备驱动 Makefile
# 实验二：Linux网络设备驱动开发
# =============================================================================

# 驱动模块名称
obj-m := netdrv.o
netdrv-objs := vnetdev.o

# 内核源码路径 - 根据实际环境修改
# 方法1: 自动检测当前运行内核的路径
KERNELDIR ?= /lib/modules/$(shell uname -r)/build

# 方法2: 手动指定（如果自动检测不工作，取消注释并修改为实际路径）
# KERNELDIR = /usr/src/kernels/$(shell uname -r)
# KERNELDIR = /usr/src/linux-headers-$(shell uname -r)

# 当前目录
PWD := $(shell pwd)

# 编译选项
EXTRA_CFLAGS += -DDEBUG_PRINT

# 默认目标：编译驱动模块
all: modules

modules:
	@echo "=========================================="
	@echo "正在编译虚拟网络设备驱动..."
	@echo "内核路径: $(KERNELDIR)"
	@echo "=========================================="
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
	@echo "=========================================="
	@echo "编译完成! 生成 netdrv.ko"
	@echo "=========================================="

# 清理编译产物
clean:
	@echo "清理编译文件..."
	rm -rf *.o *.ko *.mod.c *.mod.o *.symvers *.order .*.cmd .tmp_versions Module.symvers modules.order
	@echo "清理完成!"

# 安装驱动（需要root权限）
install: modules
	@echo "加载驱动模块..."
	sudo insmod netdrv.ko
	@echo "启动虚拟网卡..."
	sudo ifconfig vnet0 up
	@echo "驱动已加载!"
	@echo "使用 'dmesg | tail -50' 查看驱动打印"

# 卸载驱动（需要root权限）
uninstall:
	@echo "关闭虚拟网卡..."
	-sudo ifconfig vnet0 down 2>/dev/null
	@echo "卸载驱动模块..."
	-sudo rmmod netdrv 2>/dev/null
	@echo "驱动已卸载!"

# 测试：修改MTU触发驱动
test-mtu:
	@echo "修改MTU触发驱动..."
	sudo ifconfig vnet0 mtu 1222
	@echo "查看驱动打印: dmesg | tail -20"

# 显示内核日志
log:
	@echo "最近的内核日志:"
	@echo "=========================================="
	dmesg | tail -50

# 帮助信息
help:
	@echo "使用说明:"
	@echo "  make          - 编译驱动模块"
	@echo "  make clean    - 清理编译文件"
	@echo "  make install  - 加载驱动并启动网卡"
	@echo "  make uninstall- 卸载驱动"
	@echo "  make test-mtu - 修改MTU测试"
	@echo "  make log      - 显示内核日志"
	@echo ""
	@echo "手动操作:"
	@echo "  sudo insmod netdrv.ko     - 加载驱动"
	@echo "  sudo ifconfig vnet0 up    - 启动网卡"
	@echo "  sudo ifconfig vnet0 mtu 1222 - 修改MTU"
	@echo "  sudo ifconfig vnet0 down  - 关闭网卡"
	@echo "  sudo rmmod netdrv         - 卸载驱动"
	@echo "  dmesg | tail              - 查看内核打印"

.PHONY: all modules clean install uninstall test-mtu log help
