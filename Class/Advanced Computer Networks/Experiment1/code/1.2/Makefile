#============================================================================
# Experiment 1.2: TCP Socket Communication
# Makefile for compiling TCP server and client
#============================================================================

CC = gcc
CFLAGS = -g -Wall -O2

TARGETS = tcp_server tcp_client

all: $(TARGETS)

tcp_server: tcp_server.c
	$(CC) $(CFLAGS) -o tcp_server tcp_server.c
	@echo "[OK] tcp_server compiled successfully."

tcp_client: tcp_client.c
	$(CC) $(CFLAGS) -o tcp_client tcp_client.c
	@echo "[OK] tcp_client compiled successfully."

clean:
	rm -f $(TARGETS) *.o
	@echo "[OK] Cleaned up."

help:
	@echo "Usage:"
	@echo "  make          - Build all programs"
	@echo "  make clean    - Remove compiled programs"
	@echo ""
	@echo "Run in two terminals:"
	@echo "  Terminal 1: ./tcp_server"
	@echo "  Terminal 2: ./tcp_client"

.PHONY: all clean help
