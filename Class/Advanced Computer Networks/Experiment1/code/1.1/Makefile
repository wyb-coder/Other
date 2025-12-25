#============================================================================
# Experiment 1: UDP Socket Communication
# Makefile for compiling UDP server and client
#============================================================================

# Compiler settings
CC = gcc
CFLAGS = -g -Wall -O2

# Target programs
TARGETS = udp_server udp_client

# Default target: build all
all: $(TARGETS)

# Compile UDP server
udp_server: udp_server.c
	$(CC) $(CFLAGS) -o udp_server udp_server.c
	@echo "[OK] udp_server compiled successfully."

# Compile UDP client
udp_client: udp_client.c
	$(CC) $(CFLAGS) -o udp_client udp_client.c
	@echo "[OK] udp_client compiled successfully."

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o
	@echo "[OK] Cleaned up."

# Help information
help:
	@echo "Usage:"
	@echo "  make          - Build all programs"
	@echo "  make clean    - Remove compiled programs"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "After building, run in two terminals:"
	@echo "  Terminal 1: ./udp_server"
	@echo "  Terminal 2: ./udp_client"

.PHONY: all clean help
