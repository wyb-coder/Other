#============================================================================
# Experiment 1.4: Advanced Socket Programming (Options and Control Info)
# Makefile
#============================================================================

CC = gcc
CFLAGS = -g -Wall -O2

TARGETS = socket_options udp_control_info

all: $(TARGETS)

socket_options: socket_options.c
	$(CC) $(CFLAGS) -o socket_options socket_options.c
	@echo "[OK] socket_options compiled successfully."

udp_control_info: udp_control_info.c
	$(CC) $(CFLAGS) -o udp_control_info udp_control_info.c
	@echo "[OK] udp_control_info compiled successfully."

clean:
	rm -f $(TARGETS) *.o
	@echo "[OK] Cleaned up."

help:
	@echo "Usage:"
	@echo "  make              - Build all programs"
	@echo "  make clean        - Remove compiled programs"
	@echo ""
	@echo "Programs:"
	@echo "  ./socket_options      - Show and set socket options"
	@echo "  ./udp_control_info    - Demo control info with recvmsg"

.PHONY: all clean help
