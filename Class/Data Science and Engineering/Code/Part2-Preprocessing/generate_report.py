import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# Configuration
# ============================================================================
ORIGINAL_FILE = "../../Data/82.csv"
PROCESSED_FILE = "../../Data/82_processed.csv"
REPORT_FILE = "../../Data/processing_stats.txt"
REPORT_COPY = "./processing_stats.txt"  # Copy in Part2-Preprocessing

def generate_comprehensive_report():
    """Generate detailed statistics report for thesis writing."""
    
    print("Loading data files...")
    original_df = pd.read_csv(ORIGINAL_FILE)
    processed_df = pd.read_csv(PROCESSED_FILE)
    
    original_df['data_time'] = pd.to_datetime(original_df['data_time'])
    processed_df['data_time'] = pd.to_datetime(processed_df['data_time'])
    
    original_df['day'] = original_df['data_time'].dt.day
    original_df['hour'] = original_df['data_time'].dt.hour
    original_df['day_name'] = original_df['data_time'].dt.day_name()
    
    processed_df['day'] = processed_df['data_time'].dt.day
    processed_df['hour'] = processed_df['data_time'].dt.hour
    processed_df['day_name'] = processed_df['data_time'].dt.day_name()
    
    friday_orig = original_df[original_df['day'] == 1]
    saturday_orig = original_df[original_df['day'] == 2]
    
    report = []
    
    report.append("=" * 80)
    report.append("Phase 2: 数据预处理与扩增 - 完整统计报告")
    report.append("Distribution-Aware Data Augmentation Statistics Report")
    report.append("=" * 80)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"原始文件: {ORIGINAL_FILE}")
    report.append(f"处理后文件: {PROCESSED_FILE}")
    
    # ========== Section 1: Data Cleaning Summary ==========
    report.append("\n" + "=" * 80)
    report.append("1. 数据清洗摘要 (Data Cleaning Summary)")
    report.append("=" * 80)
    report.append(f"\n原始数据行数: {len(original_df):,}")
    report.append(f"清洗后数据行数: {len(processed_df[processed_df['day'] <= 2]):,}")
    report.append(f"删除/合并的记录数: {len(original_df) - len(processed_df[processed_df['day'] <= 2]):,}")
    report.append(f"\n清洗规则:")
    report.append(f"  - 删除 volume < 0 的记录")
    report.append(f"  - 删除 speed < 0 或 speed > 150 的记录")
    report.append(f"  - 合并同一监测点同一时间戳的重复记录")
    
    # ========== Section 2: Distribution Learning ==========
    report.append("\n" + "=" * 80)
    report.append("2. 分布学习结果 (Distribution Learning Results)")
    report.append("=" * 80)
    
    report.append("\n2.1 周五 (工作日模板) 分布特征:")
    report.append(f"  总记录数: {len(friday_orig):,}")
    report.append(f"  总流量: {friday_orig['volume'].sum():,}")
    report.append(f"  流量均值: {friday_orig['volume'].mean():.2f}")
    report.append(f"  流量标准差: {friday_orig['volume'].std():.2f}")
    report.append(f"  速度均值: {friday_orig['speed'].mean():.2f} km/h")
    report.append(f"  速度标准差: {friday_orig['speed'].std():.2f}")
    
    report.append("\n2.2 周六 (休息日模板) 分布特征:")
    report.append(f"  总记录数: {len(saturday_orig):,}")
    report.append(f"  总流量: {saturday_orig['volume'].sum():,}")
    report.append(f"  流量均值: {saturday_orig['volume'].mean():.2f}")
    report.append(f"  流量标准差: {saturday_orig['volume'].std():.2f}")
    report.append(f"  速度均值: {saturday_orig['speed'].mean():.2f} km/h")
    report.append(f"  速度标准差: {saturday_orig['speed'].std():.2f}")
    
    # Comparison
    fri_total = friday_orig['volume'].sum()
    sat_total = saturday_orig['volume'].sum()
    report.append("\n2.3 工作日 vs 休息日 对比:")
    report.append(f"  周六/周五总流量比: {sat_total/fri_total:.4f}")
    report.append(f"  实际差异: 周六流量约为周五的 {sat_total/fri_total*100:.1f}%")
    
    # Friday evening effect
    fri_evening = friday_orig[friday_orig['hour'] >= 17]['volume'].sum()
    sat_evening = saturday_orig[saturday_orig['hour'] >= 17]['volume'].sum()
    fri_morning = friday_orig[(friday_orig['hour'] >= 7) & (friday_orig['hour'] <= 9)]['volume'].sum()
    sat_morning = saturday_orig[(saturday_orig['hour'] >= 7) & (saturday_orig['hour'] <= 9)]['volume'].sum()
    
    report.append("\n2.4 时段分析 (验证校园交通特征):")
    report.append(f"  [早高峰 7-9时]")
    report.append(f"    周五: {fri_morning:,}")
    report.append(f"    周六: {sat_morning:,}")
    report.append(f"    周五/周六比: {fri_morning/sat_morning:.2f}x")
    report.append(f"  [晚间 17时后]")
    report.append(f"    周五: {fri_evening:,}")
    report.append(f"    周六: {sat_evening:,}")
    report.append(f"    周五/周六比: {fri_evening/sat_evening:.2f}x ← 验证了周五放学效应")
    
    # ========== Section 3: Hourly Distribution Table ==========
    report.append("\n" + "=" * 80)
    report.append("3. 小时级流量分布 (Hourly Volume Distribution)")
    report.append("=" * 80)
    report.append("\n小时 | 周五流量 | 周六流量 | 周六/周五比")
    report.append("-" * 50)
    
    for hour in range(24):
        fri_h = friday_orig[friday_orig['hour'] == hour]['volume'].sum()
        sat_h = saturday_orig[saturday_orig['hour'] == hour]['volume'].sum()
        ratio = sat_h / fri_h if fri_h > 0 else 0
        report.append(f"  {hour:02d}  |  {fri_h:6,}  |  {sat_h:6,}  |   {ratio:.3f}")
    
    # ========== Section 4: Augmentation Methodology ==========
    report.append("\n" + "=" * 80)
    report.append("4. 数据扩增方法论 (Augmentation Methodology)")
    report.append("=" * 80)
    report.append("\n4.1 扩增策略: 分布感知的跨领域数据增强")
    report.append("    Strategy: Distribution-Aware Cross-Domain Augmentation")
    report.append("\n4.2 核心参数:")
    report.append("    - 周五晚间效应系数 (Friday Evening Coefficient): β = 1.15")
    report.append("    - 周五晚间起始时间: 17:00")
    report.append("    - 高斯噪声均值: μ = 0")
    report.append("    - 高斯噪声标准差: σ = 0.03 (3%)")
    report.append("\n4.3 模板选择规则:")
    report.append("    | 目标日期 | 星期 | 类型 | 模板来源 |")
    report.append("    |----------|------|------|----------|")
    report.append("    | 03-01    | 周五 | 工作日 | 原始数据 |")
    report.append("    | 03-02    | 周六 | 休息日 | 原始数据 |")
    report.append("    | 03-03    | 周日 | 休息日 | 周六分布采样 |")
    report.append("    | 03-04    | 周一 | 工作日 | 周五分布采样 |")
    report.append("    | 03-05    | 周二 | 工作日 | 周五分布采样 |")
    report.append("    | 03-06    | 周三 | 工作日 | 周五分布采样 |")
    report.append("    | 03-07    | 周四 | 工作日 | 周五分布采样 |")
    
    # ========== Section 5: Augmentation Results ==========
    report.append("\n" + "=" * 80)
    report.append("5. 扩增结果统计 (Augmentation Results)")
    report.append("=" * 80)
    
    report.append(f"\n5.1 数据规模:")
    report.append(f"    扩增前: {len(original_df):,} 行 (2 天)")
    report.append(f"    扩增后: {len(processed_df):,} 行 (7 天)")
    report.append(f"    扩充倍数: {len(processed_df)/len(original_df):.2f}x")
    
    report.append(f"\n5.2 各日数据量:")
    for day in range(1, 8):
        day_data = processed_df[processed_df['day'] == day]
        day_name = day_data['day_name'].iloc[0] if len(day_data) > 0 else "N/A"
        report.append(f"    03-0{day} ({day_name}): {len(day_data):,} 行")
    
    report.append(f"\n5.3 扩增后流量统计:")
    report.append(f"    均值: {processed_df['volume'].mean():.2f}")
    report.append(f"    标准差: {processed_df['volume'].std():.2f}")
    report.append(f"    最小值: {processed_df['volume'].min()}")
    report.append(f"    最大值: {processed_df['volume'].max()}")
    
    report.append(f"\n5.4 扩增后速度统计:")
    report.append(f"    均值: {processed_df['speed'].mean():.2f} km/h")
    report.append(f"    标准差: {processed_df['speed'].std():.2f}")
    report.append(f"    最小值: {processed_df['speed'].min()}")
    report.append(f"    最大值: {processed_df['speed'].max()}")
    
    # ========== Section 6: Distribution Preservation Verification ==========
    report.append("\n" + "=" * 80)
    report.append("6. 分布保持性验证 (Distribution Preservation Verification)")
    report.append("=" * 80)
    
    # Compare original Friday vs generated workdays
    orig_fri_mean = friday_orig['volume'].mean()
    orig_fri_std = friday_orig['volume'].std()
    
    gen_workdays = processed_df[processed_df['day'].isin([4, 5, 6, 7])]  # Mon-Thu
    gen_work_mean = gen_workdays['volume'].mean()
    gen_work_std = gen_workdays['volume'].std()
    
    report.append("\n6.1 工作日分布保持性:")
    report.append(f"    原始周五均值: {orig_fri_mean:.2f}")
    report.append(f"    生成工作日均值: {gen_work_mean:.2f}")
    report.append(f"    均值偏差: {abs(orig_fri_mean - gen_work_mean) / orig_fri_mean * 100:.2f}%")
    report.append(f"    原始周五标准差: {orig_fri_std:.2f}")
    report.append(f"    生成工作日标准差: {gen_work_std:.2f}")
    
    # Compare original Saturday vs generated Sunday
    orig_sat_mean = saturday_orig['volume'].mean()
    orig_sat_std = saturday_orig['volume'].std()
    
    gen_sunday = processed_df[processed_df['day'] == 3]
    gen_sun_mean = gen_sunday['volume'].mean()
    gen_sun_std = gen_sunday['volume'].std()
    
    report.append("\n6.2 休息日分布保持性:")
    report.append(f"    原始周六均值: {orig_sat_mean:.2f}")
    report.append(f"    生成周日均值: {gen_sun_mean:.2f}")
    report.append(f"    均值偏差: {abs(orig_sat_mean - gen_sun_mean) / orig_sat_mean * 100:.2f}%")
    report.append(f"    原始周六标准差: {orig_sat_std:.2f}")
    report.append(f"    生成周日标准差: {gen_sun_std:.2f}")
    
    # ========== Footer ==========
    report.append("\n" + "=" * 80)
    report.append("报告生成完毕 / Report Generated Successfully")
    report.append("=" * 80)
    
    # Save report
    report_text = '\n'.join(report)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Report saved to: {REPORT_FILE}")
    
    with open(REPORT_COPY, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Report copied to: {REPORT_COPY}")
    
    return report_text

if __name__ == "__main__":
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    report = generate_comprehensive_report()
    print("\n" + "=" * 60)
    print("Comprehensive report generated successfully!")
    print("=" * 60)
