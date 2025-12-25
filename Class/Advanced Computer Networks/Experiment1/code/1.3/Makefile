#============================================================================
# Experiment 1.3: RAW Socket Communication
# Makefile for compiling RAW socket programs
#============================================================================

CC = gcc
CFLAGS = -g -Wall -O2

TARGETS = raw_receiver raw_sender

all: $(TARGETS)

raw_receiver: raw_receiver.c
	$(CC) $(CFLAGS) -o raw_receiver raw_receiver.c
	@echo "[OK] raw_receiver compiled successfully."

raw_sender: raw_sender.c
	$(CC) $(CFLAGS) -o raw_sender raw_sender.c
	@echo "[OK] raw_sender compiled successfully."

clean:
	rm -f $(TARGETS) *.o
	@echo "[OK] Cleaned up."

help:
	@echo "Usage:"
	@echo "  make          - Build all programs"
	@echo "  make clean    - Remove compiled programs"
	@echo ""
	@echo "NOTE: RAW sockets require root privilege!"
	@echo "  sudo ./raw_receiver   - Receive ICMP packets"
	@echo "  sudo ./raw_sender     - Send ICMP Echo Requests"

.PHONY: all clean help
