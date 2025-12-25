#============================================================================
# Experiment 1.5: Multi-threaded Server
# Makefile
#============================================================================

CC = gcc
CFLAGS = -g -Wall -O2 -pthread

TARGETS = mt_server mt_client

all: $(TARGETS)

mt_server: mt_server.c
	$(CC) $(CFLAGS) -o mt_server mt_server.c
	@echo "[OK] mt_server compiled successfully."

mt_client: mt_client.c
	$(CC) $(CFLAGS) -o mt_client mt_client.c
	@echo "[OK] mt_client compiled successfully."

clean:
	rm -f $(TARGETS) *.o
	@echo "[OK] Cleaned up."

help:
	@echo "Usage:"
	@echo "  make          - Build all programs"
	@echo "  make clean    - Remove compiled programs"
	@echo ""
	@echo "Run:"
	@echo "  Terminal 1: ./mt_server"
	@echo "  Terminal 2: ./mt_client  (or multiple nc connections)"

.PHONY: all clean help
