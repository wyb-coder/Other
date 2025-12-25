from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, avg, stddev, count, sum as spark_sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import csv

# Configuration
DATA_FILE = "file:///opt/spark/work-dir/82_processed.csv"
OUTPUT_CSV = "/opt/spark/work-dir/daily_correlation_spark.csv"

def init_spark():
    return SparkSession.builder \
        .appName("TrafficCorrelation_Daily_SQL") \
        .getOrCreate()

def pearson_corr(spark, df, col_a, col_b, alias_a, alias_b, date_str):
    """
    Calculate Pearson correlation coefficient between two columns using SQL.
    Formula: r = Σ[(xi - x̄)(yi - ȳ)] / [√Σ(xi - x̄)² * √Σ(yi - ȳ)²]
    Simplified: r = [n*Σxy - Σx*Σy] / √[(n*Σx² - (Σx)²)(n*Σy² - (Σy)²)]
    """
    stats = df.agg(
        count("*").alias("n"),
        spark_sum(col(col_a)).alias("sum_x"),
        spark_sum(col(col_b)).alias("sum_y"),
        spark_sum(col(col_a) * col(col_a)).alias("sum_x2"),
        spark_sum(col(col_b) * col(col_b)).alias("sum_y2"),
        spark_sum(col(col_a) * col(col_b)).alias("sum_xy")
    ).collect()[0]
    
    n = stats["n"]
    sum_x = stats["sum_x"] or 0
    sum_y = stats["sum_y"] or 0
    sum_x2 = stats["sum_x2"] or 0
    sum_y2 = stats["sum_y2"] or 0
    sum_xy = stats["sum_xy"] or 0
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator_x = n * sum_x2 - sum_x * sum_x
    denominator_y = n * sum_y2 - sum_y * sum_y
    
    if denominator_x <= 0 or denominator_y <= 0:
        return 0.0
    
    denominator = (denominator_x ** 0.5) * (denominator_y ** 0.5)
    
    if denominator == 0:
        return 0.0
    
    r = numerator / denominator
    return r

def main():
    spark = init_spark()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"Loading data from: {DATA_FILE}")
    
    df = spark.read.csv(DATA_FILE, header=True, inferSchema=True)
    
    # Convert data_time to timestamp and get date
    df = df.withColumn("timestamp", to_timestamp(col("data_time"))) \
           .withColumn("date", date_format(col("timestamp"), "yyyy-MM-dd"))

    # Get list of dates to iterate
    dates_row = df.select("date").distinct().sort("date").collect()
    dates = [row.date for row in dates_row if row.date is not None]
    
    print(f"Found {len(dates)} days: {dates}")
    
    # Get unique road segment IDs
    unique_ids_row = df.select("road_seg_id").distinct().sort("road_seg_id").collect()
    unique_ids = [row.road_seg_id for row in unique_ids_row]
    id_map = {uid: f"S{i+1}" for i, uid in enumerate(unique_ids)}
    
    all_results = []

    for process_date in dates:
        print(f"\nProcessing {process_date}...")
        
        day_df = df.filter(col("date") == process_date)
        
        # Pivot: Row=Time, Col=Road, Val=Volume
        pivoted = day_df.groupBy("data_time") \
                        .pivot("road_seg_id", values=unique_ids) \
                        .sum("volume") \
                        .na.fill(0)
        
        # Calculate pairwise correlations
        for i in range(len(unique_ids)):
            for j in range(i+1, len(unique_ids)):
                id_a = unique_ids[i]
                id_b = unique_ids[j]
                
                r = pearson_corr(spark, pivoted, id_a, id_b, id_map[id_a], id_map[id_b], process_date)
                
                all_results.append({
                    "Monitor_a": id_map[id_a],
                    "Monitor_b": id_map[id_b],
                    "Date": process_date,
                    "Correlation": r,
                    "Monitor_a_Full": id_a,
                    "Monitor_b_Full": id_b
                })

    # Sort by Correlation (Descending)
    all_results.sort(key=lambda x: x["Correlation"], reverse=True)
    
    # Write CSV using built-in csv module
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Monitor_a", "Monitor_b", "Date", "Correlation", "Monitor_a_Full", "Monitor_b_Full"])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nSaved ranked correlations to {OUTPUT_CSV}")
    
    print("\nTop 5 Correlations:")
    for row in all_results[:5]:
        print(f"  {row['Monitor_a']} - {row['Monitor_b']} ({row['Date']}): {row['Correlation']:.4f}")
    
    spark.stop()

if __name__ == "__main__":
    main()
