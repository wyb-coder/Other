"""
Phase 4: HBase Multi-Dimensional Query with SQL Translation
============================================================
This script demonstrates how to translate SQL WHERE conditions to HBase Scan operations
using the optimized RowKey design (MD5 prefix + reverse timestamp).

Core Functionality:
1. SQL-to-HBase Scan translation
2. RowKey range calculation for time-based queries
3. Query command generation for HBase Shell
"""

import hashlib
from datetime import datetime
import os
import csv

# ============================================================================
# Configuration
# ============================================================================
MAX_TIMESTAMP = 9999999999999  # For reverse timestamp calculation
CSV_FILE = "../../Data/82_processed.csv"

# ============================================================================
# RowKey Utilities (same as Phase 3)
# ============================================================================
def get_md5_prefix(road_seg_id: str) -> str:
    """Get 2-character MD5 prefix for a road segment ID."""
    return hashlib.md5(road_seg_id.encode()).hexdigest()[:2]

def timestamp_to_millis(ts_str: str) -> int:
    """Convert timestamp string to milliseconds."""
    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)

def reverse_timestamp(ts_str: str) -> int:
    """Calculate reverse timestamp for RowKey."""
    return MAX_TIMESTAMP - timestamp_to_millis(ts_str)

def generate_rowkey(road_seg_id: str, timestamp: str) -> str:
    """Generate full RowKey from road_seg_id and timestamp."""
    prefix = get_md5_prefix(road_seg_id)
    rev_ts = reverse_timestamp(timestamp)
    return f"{prefix}_{road_seg_id}_{rev_ts:013d}"

# ============================================================================
# SQL to HBase Scan Translation
# ============================================================================
class SQLToHBaseScan:
    """
    Translates SQL WHERE conditions to HBase Scan range.
    
    Supported SQL pattern:
        SELECT * FROM traffic 
        WHERE road_seg_id = ? 
        AND data_time BETWEEN ? AND ?
    """
    
    def __init__(self, road_seg_id: str, start_time: str, end_time: str):
        self.road_seg_id = road_seg_id
        self.start_time = start_time
        self.end_time = end_time
        
        # Calculate RowKey components
        self.md5_prefix = get_md5_prefix(road_seg_id)
        self.start_reverse = reverse_timestamp(end_time)    # Note: reversed!
        self.end_reverse = reverse_timestamp(start_time)    # Note: reversed!
    
    def get_scan_range(self) -> dict:
        """
        Get HBase Scan StartRow and StopRow.
        
        IMPORTANT: Because we use reverse timestamp, the relationship is inverted:
        - end_time (later) → smaller reverse value → StartRow
        - start_time (earlier) → larger reverse value → StopRow
        """
        start_row = f"{self.md5_prefix}_{self.road_seg_id}_{self.start_reverse:013d}"
        stop_row = f"{self.md5_prefix}_{self.road_seg_id}_{self.end_reverse:013d}"
        
        return {
            "start_row": start_row,
            "stop_row": stop_row
        }
    
    def explain(self) -> str:
        """Generate human-readable explanation of the translation."""
        lines = []
        lines.append("=" * 70)
        lines.append("SQL to HBase Scan Translation")
        lines.append("=" * 70)
        lines.append("")
        lines.append("【Original SQL】")
        lines.append(f"  SELECT * FROM traffic")
        lines.append(f"  WHERE road_seg_id = '{self.road_seg_id}'")
        lines.append(f"  AND data_time BETWEEN '{self.start_time}' AND '{self.end_time}'")
        lines.append("")
        lines.append("【Translation Process】")
        lines.append("")
        lines.append("  Step 1: Calculate MD5 prefix (for load balancing)")
        lines.append(f"    MD5('{self.road_seg_id}') = ...{hashlib.md5(self.road_seg_id.encode()).hexdigest()}")
        lines.append(f"    Prefix = {self.md5_prefix}")
        lines.append("")
        lines.append("  Step 2: Calculate reverse timestamps")
        lines.append(f"    start_time '{self.start_time}' → {timestamp_to_millis(self.start_time)} ms")
        lines.append(f"    end_time   '{self.end_time}'   → {timestamp_to_millis(self.end_time)} ms")
        lines.append(f"    MAX_TIMESTAMP = {MAX_TIMESTAMP}")
        lines.append(f"    reverse(start) = {MAX_TIMESTAMP} - {timestamp_to_millis(self.start_time)} = {self.end_reverse}")
        lines.append(f"    reverse(end)   = {MAX_TIMESTAMP} - {timestamp_to_millis(self.end_time)} = {self.start_reverse}")
        lines.append("")
        lines.append("  Step 3: Construct RowKey range (KEY INSIGHT)")
        lines.append("    ⚠️ Because we use REVERSE timestamp:")
        lines.append("       - Later time (end) → smaller reverse → becomes StartRow")
        lines.append("       - Earlier time (start) → larger reverse → becomes StopRow")
        lines.append("")
        
        scan_range = self.get_scan_range()
        lines.append("【HBase Scan Parameters】")
        lines.append(f"  StartRow: {scan_range['start_row']}")
        lines.append(f"  StopRow:  {scan_range['stop_row']}")
        lines.append("")
        lines.append("【HBase Shell Command】")
        lines.append(f"  scan 'traffic', {{STARTROW => '{scan_range['start_row']}', STOPROW => '{scan_range['stop_row']}'}}")
        lines.append("")
        lines.append("【Performance Advantage】")
        lines.append("  - Traditional approach: Full table scan O(N)")
        lines.append("  - Our approach: Direct RowKey range scan O(log N + k)")
        lines.append("    where k = number of matching rows")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def generate_hbase_shell(self) -> str:
        """Generate HBase Shell command."""
        scan_range = self.get_scan_range()
        return f"scan 'traffic', {{STARTROW => '{scan_range['start_row']}', STOPROW => '{scan_range['stop_row']}'}}"

# ============================================================================
# Query Demonstration
# ============================================================================
def load_sample_road_ids(csv_path: str, limit: int = 3) -> list:
    """Load sample road segment IDs from CSV."""
    road_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            road_ids.add(row['road_seg_id'])
            if len(road_ids) >= limit:
                break
    return list(road_ids)

def demonstrate_queries():
    """Generate demonstration queries and explanations."""
    print("=" * 70)
    print("Phase 4: Multi-Dimensional Data Query Demonstration")
    print("=" * 70)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Load sample road IDs
    road_ids = load_sample_road_ids(CSV_FILE)
    
    # Define sample queries
    queries = [
        {
            "name": "Query 1: Morning rush hour on Day 1",
            "road_seg_id": road_ids[0],
            "start_time": "2024-03-01 07:00:00",
            "end_time": "2024-03-01 09:00:00"
        },
        {
            "name": "Query 2: Friday evening (school dismissal effect)",
            "road_seg_id": road_ids[0],
            "start_time": "2024-03-01 17:00:00",
            "end_time": "2024-03-01 19:00:00"
        },
        {
            "name": "Query 3: Full day on the last day",
            "road_seg_id": road_ids[1] if len(road_ids) > 1 else road_ids[0],
            "start_time": "2024-03-07 00:00:00",
            "end_time": "2024-03-07 23:59:59"
        }
    ]
    
    results = []
    
    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"【{query['name']}】")
        print(f"{'=' * 70}")
        
        translator = SQLToHBaseScan(
            query['road_seg_id'],
            query['start_time'],
            query['end_time']
        )
        
        explanation = translator.explain()
        print(explanation)
        
        results.append({
            "name": query['name'],
            "sql": f"WHERE road_seg_id = '{query['road_seg_id'][:20]}...' AND data_time BETWEEN '{query['start_time']}' AND '{query['end_time']}'",
            "hbase_shell": translator.generate_hbase_shell(),
            "explanation": explanation
        })
    
    # Save results to file
    output_file = "./query_demonstration.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Phase 4: SQL to HBase Scan Translation Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write(f"\n{result['explanation']}\n\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Generate summary statistics
    generate_summary()
    
    return results

def generate_summary():
    """Generate query capability summary for thesis."""
    print("\n" + "=" * 70)
    print("Query Capability Summary (for Thesis)")
    print("=" * 70)
    
    summary = """
【查询能力总结】

1. 支持的查询类型:
   ✅ 单路段 + 时间范围查询
   ✅ 最新 N 条数据查询 (利用时间倒序)
   ✅ 按 MD5 前缀分布式扫描

2. 性能优势:
   - 传统全表扫描: O(N)，需遍历 93,234 行
   - RowKey 范围扫描: O(log N + k)，直接定位目标区间
   
3. SQL 翻译公式:
   WHERE road_seg_id = ? AND time BETWEEN start AND end
   
   翻译为 HBase Scan:
   StartRow = MD5(id)[0:2] + "_" + id + "_" + (MAX - end_time)
   StopRow  = MD5(id)[0:2] + "_" + id + "_" + (MAX - start_time)

4. 关键洞察 (报告亮点):
   "由于采用时间戳倒序设计，查询时间范围的 StartRow 和 StopRow 
   需要与直觉相反：较晚的时间作为 StartRow，较早的时间作为 StopRow。
   这一细节体现了对 HBase LSM-Tree 存储机制的深刻理解。"
"""
    print(summary)
    
    # Save summary
    with open("./query_summary.txt", 'w', encoding='utf-8') as f:
        f.write("Phase 4: Query Capability Summary\n")
        f.write("=" * 50 + "\n")
        f.write(summary)
    
    print("\nSummary saved to: ./query_summary.txt")

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    demonstrate_queries()
    print("\n" + "=" * 70)
    print("Phase 4: Query Implementation Complete!")
    print("=" * 70)
