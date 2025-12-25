import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
DATA_FILE = "../../Data/82_processed.csv"
OUTPUT_DIR = "../../Data"
IMG_DIR = "../../picture"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    df['data_time'] = pd.to_datetime(df['data_time'])
    df['date'] = df['data_time'].dt.date.astype(str)
    
    # Map long IDs to Short IDs (S1, S2...) for better readability in charts
    unique_ids = sorted(df['road_seg_id'].unique())
    id_map = {uid: f"S{i+1}" for i, uid in enumerate(unique_ids)}
    
    results = []
    
    # Process each day
    days = sorted(df['date'].unique())
    print(f"Found {len(days)} days: {days}")
    
    for day in days:
        print(f"\nProcessing {day}...")
        day_df = df[df['date'] == day]
        
        # Pivot: Index=Minute, Columns=Road Segment, Value=Volume
        # Handle duplicates by taking mean (though minute-level should be unique per sensor)
        pivot = day_df.pivot_table(index='data_time', columns='road_seg_id', values='volume', aggfunc='mean').fillna(0)
        
        # Calculate Correlation
        corr_matrix = pivot.corr(method='pearson')
        
        # 1. Generate Heatmap
        plt.figure(figsize=(10, 8))
        # Use Short IDs for axis labels
        short_labels = [id_map[c] for c in corr_matrix.columns]
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    xticklabels=short_labels, yticklabels=short_labels, vmin=-1, vmax=1)
        plt.title(f"Traffic Correlation Matrix - {day}")
        
        heatmap_path = f"{IMG_DIR}/correlation_heatmap_{day}.png"
        plt.savefig(heatmap_path)
        plt.close()
        print(f"  Saved heatmap to {heatmap_path}")
        
        # 2. Extract pairs
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r = corr_matrix.iloc[i, j]
                results.append({
                    "Monitor_a": id_map[cols[i]], # Using Short ID as per likely requirement for readability, or use cols[i] for full ID
                    "Monitor_b": id_map[cols[j]],
                    "Monitor_a_Full": cols[i],
                    "Monitor_b_Full": cols[j],
                    "Date": day,
                    "Correlation": r
                })

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Sort by Correlation (Descending) - As per requirement "按照相关性大小排序输出"
    # Usually absolute correlation matters, but the example table shows 0.98, 0.87. 
    # If there are negative correlations, sorting by simple value might put -0.99 at bottom.
    # The requirement says "size" (大小), which implies magnitude, or just descending algebraic value?
    # Given typical traffic data is mostly positive correlation, descending algebraic is fine.
    # But to be safe for "Magnitude", I will verify if we have negatives.
    # Let's sort by value descending for now.
    res_df = res_df.sort_values(by='Correlation', ascending=False)
    
    # Save formatted CSV
    output_csv = f"{OUTPUT_DIR}/daily_correlation_results.csv"
    res_df.to_csv(output_csv, index=False)
    print(f"\nSaved analysis results to {output_csv}")
    
    # Print sample top rows for user to see
    print("\nTop 10 Correlations:")
    print(res_df[['Monitor_a', 'Monitor_b', 'Date', 'Correlation']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
