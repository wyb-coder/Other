import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data from thesis outline
data = {
    'Date': ['03-01', '03-02', '03-03', '03-04', '03-05', '03-06', '03-07'],
    'Day': ['周五', '周六', '周日', '周一', '周二', '周三', '周四'],
    'Total Volume': [438563, 406246, 377350, 387841, 393209, 388001, 386889],
    'Type': ['Weekday', 'Weekend', 'Weekend', 'Weekday', 'Weekday', 'Weekday', 'Weekday']
}

df = pd.DataFrame(data)

# Set style
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # For Chinese characters
plt.rcParams['axes.unicode_minus'] = False

# Create figure
plt.figure(figsize=(10, 6))

# Define colors: Weekday (Blue), Weekend (Orange)
colors = ['#4c72b0' if x == 'Weekday' else '#dd8452' for x in df['Type']]

# Create bar plot
bars = plt.bar(df['Date'] + '\n' + df['Day'], df['Total Volume'], color=colors, alpha=0.9)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=10)

# Highlight Peak (Friday) and Valley (Sunday)
# Friday is index 0, Sunday is index 2
plt.annotate('Peak (峰值)', 
             xy=(bars[0].get_x() + bars[0].get_width()/2, df['Total Volume'][0]), 
             xytext=(0, 20), textcoords='offset points',
             ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'),
             fontsize=11, color='red', fontweight='bold')

plt.annotate('Valley (谷值)', 
             xy=(bars[2].get_x() + bars[2].get_width()/2, df['Total Volume'][2]), 
             xytext=(0, 20), textcoords='offset points',
             ha='center', va='bottom',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='green'),
             fontsize=11, color='green', fontweight='bold')

# Labels and Title
plt.title('7-Day Aggregated Traffic Volume Trend (Workday vs Weekend)', fontsize=14, pad=20)
plt.ylabel('Total Traffic Volume', fontsize=12)
plt.xlabel('Date', fontsize=12)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4c72b0', label='Weekday (工作日)'),
                   Patch(facecolor='#dd8452', label='Weekend (休息日)')]
plt.legend(handles=legend_elements, loc='upper right')

# Adjust layout
plt.tight_layout()

# Save
output_path = 'd:\\Study\\CS Study\\Code\\Other\\Class\\Data Science and Engineering\\picture\\fig5_weekly_traffic.png'
plt.savefig(output_path, dpi=300)
print(f"Chart saved to {output_path}")
