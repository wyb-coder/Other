"""
Phase 3: HBase Data Ingestion with Optimized RowKey Design
===========================================================
This script ingests traffic data from CSV into HBase with:
1. MD5 prefix for load balancing (avoiding hotspots)
2. Reverse timestamp for efficient "latest first" queries

RowKey Design: MD5(road_seg_id)[0:2] + road_seg_id + (MAX_TS - timestamp)
Example: "a3" + "13G6S0C4IE013G6P0C4KS00300772" + "9999999999999-1709251200000"
"""

import csv
import hashlib
from datetime import datetime
import os

# ============================================================================
# Configuration
# ============================================================================
CSV_FILE = "../../Data/82_processed.csv"
HBASE_HOST = "localhost"
HBASE_PORT = 9090  # Thrift port

# RowKey design constants
MAX_TIMESTAMP = 9999999999999  # Max 13-digit timestamp for reverse ordering

# ============================================================================
# RowKey Design Functions
# ============================================================================
def generate_rowkey(road_seg_id: str, timestamp_str: str) -> str:
    """
    Generate optimized RowKey for HBase.
    
    Design: MD5(id)[0:2] + id + (MAX - timestamp)
    
    Purpose:
    - MD5 prefix: Distributes writes across regions (load balancing)
    - Original ID: Enables prefix scans for specific road segments
    - Reverse timestamp: Latest data has smallest suffix, appears first in scan
    
    Args:
        road_seg_id: Road segment identifier (e.g., "13G6S0C4IE013G6P0C4KS00300772")
        timestamp_str: Timestamp string (e.g., "2024-03-01 00:00:00")
    
    Returns:
        RowKey string for HBase
    """
    # Calculate MD5 prefix (first 2 characters)
    md5_hash = hashlib.md5(road_seg_id.encode()).hexdigest()
    md5_prefix = md5_hash[:2]
    
    # Parse timestamp and convert to milliseconds
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    timestamp_ms = int(dt.timestamp() * 1000)
    
    # Reverse timestamp for latest-first ordering
    reverse_ts = MAX_TIMESTAMP - timestamp_ms
    
    # Construct RowKey
    rowkey = f"{md5_prefix}_{road_seg_id}_{reverse_ts:013d}"
    
    return rowkey

def parse_rowkey(rowkey: str) -> dict:
    """Parse a RowKey back to its components (for verification)."""
    parts = rowkey.split("_")
    md5_prefix = parts[0]
    road_seg_id = parts[1]
    reverse_ts = int(parts[2])
    
    # Convert reverse timestamp back to actual timestamp
    actual_ts = MAX_TIMESTAMP - reverse_ts
    dt = datetime.fromtimestamp(actual_ts / 1000)
    
    return {
        "md5_prefix": md5_prefix,
        "road_seg_id": road_seg_id,
        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S")
    }

# ============================================================================
# Data Ingestion (Generate HBase Commands)
# ============================================================================
def generate_hbase_commands(csv_path: str, output_path: str, limit: int = None):
    """
    Generate HBase shell commands from CSV for data ingestion.
    
    Since happybase requires Thrift server (extra setup), we generate
    HBase shell commands that can be executed directly.
    """
    print("=" * 60)
    print("Phase 3: Generating HBase Ingestion Commands")
    print("=" * 60)
    
    commands = []
    commands.append("# HBase Shell Commands for Data Ingestion")
    commands.append("# Run with: hbase shell < hbase_commands.txt")
    commands.append("")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            if limit and count >= limit:
                break
            
            rowkey = generate_rowkey(row['road_seg_id'], row['data_time'])
            
            # Generate put command
            # put 'traffic', 'rowkey', 'cf:volume', 'value'
            cmd_volume = f"put 'traffic', '{rowkey}', 'cf:volume', '{row['volume']}'"
            cmd_speed = f"put 'traffic', '{rowkey}', 'cf:speed', '{row['speed']}'"
            cmd_id = f"put 'traffic', '{rowkey}', 'cf:road_seg_id', '{row['road_seg_id']}'"
            cmd_time = f"put 'traffic', '{rowkey}', 'cf:data_time', '{row['data_time']}'"
            
            commands.append(cmd_volume)
            commands.append(cmd_speed)
            commands.append(cmd_id)
            commands.append(cmd_time)
            
            count += 1
            
            if count % 10000 == 0:
                print(f"  Processed {count} rows...")
    
    # Write commands to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(commands))
    
    print(f"\nGenerated {count} row insertions ({count * 4} put commands)")
    print(f"Commands saved to: {output_path}")
    
    return count

def demonstrate_rowkey_design():
    """Demonstrate the RowKey design with examples."""
    print("\n" + "=" * 60)
    print("RowKey Design Demonstration")
    print("=" * 60)
    
    examples = [
        ("13G6S0C4IE013G6P0C4KS00300772", "2024-03-01 00:00:00"),
        ("13G6S0C4IE013G6P0C4KS00300772", "2024-03-01 00:01:00"),
        ("13G730C5JR013G9G0C5LM00100683", "2024-03-01 00:00:00"),
        ("13GQN0C644013GOB0C64300102661", "2024-03-07 23:59:00"),
    ]
    
    print("\n| Road Segment ID (truncated) | Timestamp | RowKey |")
    print("|" + "-" * 29 + "|" + "-" * 21 + "|" + "-" * 50 + "|")
    
    for road_id, ts in examples:
        rowkey = generate_rowkey(road_id, ts)
        print(f"| ...{road_id[-8:]} | {ts} | {rowkey[:45]}... |")
    
    print("\n" + "=" * 60)
    print("RowKey Properties Analysis")
    print("=" * 60)
    
    # Show how MD5 distributes different IDs
    road_ids = [
        "13G6S0C4IE013G6P0C4KS00300772",
        "13G730C5JR013G9G0C5LM00100683",
        "13GQN0C644013GOB0C64300102661",
        "13H000C647013H1N0C64600300543",
        "13H1Q0C5R8013H1R0C5Q900101614",
    ]
    
    print("\nMD5 Prefix Distribution (Load Balancing):")
    for road_id in road_ids:
        md5_hash = hashlib.md5(road_id.encode()).hexdigest()
        print(f"  {road_id[-15:]} → {md5_hash[:2]}")
    
    # Show reverse timestamp ordering
    print("\nReverse Timestamp Ordering (Latest First):")
    ts_examples = [
        "2024-03-01 00:00:00",
        "2024-03-03 12:00:00",
        "2024-03-07 23:59:00",
    ]
    road_id = "13G6S0C4IE013G6P0C4KS00300772"
    
    rowkeys = [(ts, generate_rowkey(road_id, ts)) for ts in ts_examples]
    rowkeys_sorted = sorted(rowkeys, key=lambda x: x[1])
    
    print("  Sorted by RowKey (scan order):")
    for ts, rk in rowkeys_sorted:
        print(f"    {ts} → ...{rk[-20:]}")
    print("  → Notice: Latest timestamp (03-07) appears FIRST in scan!")

# ============================================================================
# Main
# ============================================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n" + "=" * 60)
    print("Phase 3: HBase Data Ingestion")
    print("=" * 60)
    
    # Demonstrate RowKey design
    demonstrate_rowkey_design()
    
    # Generate commands for first 1000 rows (for demonstration)
    # Full ingestion would be 93,234 rows × 4 columns = 372,936 puts
    print("\n")
    generate_hbase_commands(
        CSV_FILE, 
        "./hbase_commands_sample.txt",
        limit=1000
    )
    
    # Also generate full commands file
    generate_hbase_commands(
        CSV_FILE,
        "./hbase_commands_full.txt",
        limit=None
    )
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy hbase_commands_sample.txt to HBase container")
    print("2. Execute: docker cp hbase_commands_sample.txt hbase-master:/tmp/")
    print("3. Run: docker exec hbase-master hbase shell < /tmp/hbase_commands_sample.txt")

if __name__ == "__main__":
    main()
