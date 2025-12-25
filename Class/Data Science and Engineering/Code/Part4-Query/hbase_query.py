# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
import hashlib
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
HDFS_INPUT = "file:///opt/spark/work-dir/82_processed.csv"

# ============================================================================
# Core: SQL to HBase RowKey Translation
# ============================================================================
class SQLToHBaseScan:
    """Translates SQL WHERE to HBase Scan range using Salted-Reverse-Timestamp design."""
    
    MAX_TIMESTAMP = 9999999999999
    
    def __init__(self, road_seg_id: str, start_time: str, end_time: str):
        self.road_seg_id = road_seg_id
        self.start_time = start_time
        self.end_time = end_time
        self.md5_prefix = hashlib.md5(road_seg_id.encode()).hexdigest()[:2]
        # Reverse: End time -> smaller value -> StartRow
        self.start_reverse = self._reverse_ts(end_time)
        self.end_reverse = self._reverse_ts(start_time)
    
    def _reverse_ts(self, time_str: str) -> int:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return self.MAX_TIMESTAMP - int(dt.timestamp() * 1000)

    def get_scan_range(self) -> dict:
        return {
            "start_row": f"{self.md5_prefix}_{self.road_seg_id}_{self.start_reverse:013d}",
            "stop_row":  f"{self.md5_prefix}_{self.road_seg_id}_{self.end_reverse:013d}"
        }

# ============================================================================
# Main
# ============================================================================
def main():
    # Use user's exact example parameters
    TARGET_ID = "13EQB0C3DT013EPP0C3C600104214"
    START_TIME = "2024-03-01 08:00:00"
    END_TIME = "2024-03-01 09:00:00"
    
    print("=" * 70)
    print("Phase 4: HBase Distributed Query System")
    print("=" * 70)
    
    # Step 1: Translate SQL to RowKey
    print(f"\n[Query Request]")
    print(f"  SELECT * FROM traffic")
    print(f"  WHERE road_seg_id = '{TARGET_ID}'")
    print(f"  AND data_time BETWEEN '{START_TIME}' AND '{END_TIME}'")
    
    translator = SQLToHBaseScan(TARGET_ID, START_TIME, END_TIME)
    scan = translator.get_scan_range()
    
    print(f"\n[RowKey Translation Result]")
    print(f"  StartRow: {scan['start_row']} (对应 {END_TIME})")
    print(f"  StopRow:  {scan['stop_row']} (对应 {START_TIME})")
    
    # Step 2: Execute Spark Query
    print(f"\n[Executing Spark Query...]")
    spark = SparkSession.builder.appName("HBaseQuery").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    df = spark.read.csv(HDFS_INPUT, header=True, inferSchema=True)
    df = df.withColumn("data_time", to_timestamp(col("data_time")))
    
    result = df.filter(
        (col("road_seg_id") == TARGET_ID) &
        (col("data_time") >= START_TIME) &
        (col("data_time") <= END_TIME)
    ).orderBy("data_time")
    
    count = result.count()
    print(f"\n[Query Result]")
    print(f"  Found {count} records matching the criteria")
    print(f"\n  Sample Data:")
    result.show(5, truncate=False)
    
    print("=" * 70)
    print("Query Complete!")
    print("=" * 70)
    spark.stop()

if __name__ == "__main__":
    main()
