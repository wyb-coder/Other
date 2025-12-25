# 触发程序 Makefile

CC = gcc
CFLAGS = -g -Wall

TARGET = raw_trigger

all: $(TARGET)

$(TARGET): raw_trigger.c
	$(CC) $(CFLAGS) -o $(TARGET) raw_trigger.c
	@echo "[OK] $(TARGET) 编译成功"

clean:
	rm -f $(TARGET)
	@echo "[OK] 清理完成"

# 运行触发程序（需要root权限）
run: $(TARGET)
	sudo ./$(TARGET)

help:
	@echo "使用说明:"
	@echo "  make      - 编译触发程序"
	@echo "  make run  - 运行触发程序 (需要sudo)"
	@echo "  make clean- 清理"

.PHONY: all clean run help
