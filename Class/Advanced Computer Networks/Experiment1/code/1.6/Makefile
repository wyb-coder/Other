#============================================================================
# Experiment 1.6: Multiplexed I/O (select)
# Makefile
#============================================================================

CC = gcc
CFLAGS = -g -Wall -O2

TARGETS = select_server select_client

all: $(TARGETS)

select_server: select_server.c
	$(CC) $(CFLAGS) -o select_server select_server.c
	@echo "[OK] select_server compiled successfully."

select_client: select_client.c
	$(CC) $(CFLAGS) -o select_client select_client.c
	@echo "[OK] select_client compiled successfully."

clean:
	rm -f $(TARGETS) *.o
	@echo "[OK] Cleaned up."

help:
	@echo "Usage:"
	@echo "  make          - Build all programs"
	@echo "  make clean    - Remove compiled programs"
	@echo ""
	@echo "Run:"
	@echo "  Terminal 1: ./select_server"
	@echo "  Terminal 2: ./select_client"
	@echo ""
	@echo "Client commands:"
	@echo "  t:message  - Send via TCP"
	@echo "  u:message  - Send via UDP"

.PHONY: all clean help
