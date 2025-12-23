"""
Spark Aggregation Job for Cluster Submission
=============================================
This script is designed to run on the Spark cluster for 15-minute window aggregation.

Submission command:
  docker exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    /opt/spark-apps/spark_aggregation_job.py

Or from local with spark-submit:
  spark-submit --master spark://localhost:7077 spark_aggregation_job.py
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg, window, to_timestamp

# ============================================================================
# Configuration
# ============================================================================
HDFS_INPUT = "hdfs://namenode:9000/traffic/82_processed.csv"
HDFS_OUTPUT = "hdfs://namenode:9000/traffic/aggregated_15min"

# ============================================================================
# Main Spark Job
# ============================================================================
def main():
    # Create Spark Session
    spark = SparkSession.builder \
        .appName("TrafficAggregation") \
        .config("spark.sql.shuffle.partitions", "10") \
        .getOrCreate()
    
    print("=" * 60)
    print("Spark Traffic Aggregation Job Started")
    print("=" * 60)
    
    # Read data from HDFS
    print("\nStep 1: Reading data from HDFS...")
    df = spark.read.csv(HDFS_INPUT, header=True, inferSchema=True)
    
    print(f"  Total records: {df.count()}")
    df.printSchema()
    
    # Convert data_time to timestamp if it's string
    if df.schema["data_time"].dataType.typeName() == "string":
        df = df.withColumn("data_time", to_timestamp(col("data_time")))
    
    # Cast volume and speed to numeric
    df = df.withColumn("volume", col("volume").cast("integer"))
    df = df.withColumn("speed", col("speed").cast("double"))
    
    # 15-minute window aggregation
    print("\nStep 2: Performing 15-minute window aggregation...")
    
    result = df.groupBy(
        col("road_seg_id"),
        window(col("data_time"), "15 minutes").alias("time_window")
    ).agg(
        _sum("volume").alias("total_volume"),
        avg("speed").alias("avg_speed")
    )
    
    # Extract window start and end for cleaner output
    result = result.withColumn("window_start", col("time_window.start")) \
                   .withColumn("window_end", col("time_window.end")) \
                   .drop("time_window")
    
    # Reorder columns
    result = result.select("road_seg_id", "window_start", "window_end", 
                           "total_volume", "avg_speed")
    
    print(f"  Aggregated records: {result.count()}")
    
    # Show sample
    print("\nSample output:")
    result.show(10, truncate=False)
    
    # Save to HDFS
    print(f"\nStep 3: Saving to {HDFS_OUTPUT}...")
    result.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(HDFS_OUTPUT)
    
    print("\n" + "=" * 60)
    print("Spark Job Completed Successfully!")
    print("=" * 60)
    
    spark.stop()

if __name__ == "__main__":
    main()
