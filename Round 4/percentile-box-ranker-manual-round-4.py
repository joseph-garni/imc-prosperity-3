import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Define the sample data
sample_data = {
    1: (80, 6), 2: (50, 4), 3: (83, 7), 4: (31, 2), 5: (60, 4),
    6: (89, 8), 7: (10, 1), 8: (37, 3), 9: (70, 4), 10: (90, 10),
    11: (17, 1), 12: (40, 3), 13: (73, 4), 14: (100, 15), 15: (20, 2),
    16: (41, 3), 17: (79, 5), 18: (23, 2), 19: (47, 3), 20: (30, 2)
}

def calculate_profit(box_value, num_contestants, percentage):
    """Calculate profit based on the formula:
    profit = (box value × percentage) ÷ (percentage + number of contestants in box)
    """
    return (box_value * percentage) / (percentage + num_contestants)

# Calculate profits for percentages 1-10
percentages = range(1, 11)
all_data = []

for percentage in percentages:
    for box_num, (box_value, contestants) in sample_data.items():
        profit = calculate_profit(box_value, contestants, percentage)
        all_data.append({
            'Percentage': percentage,
            'Box': box_num,
            'Box Value': box_value,
            'Contestants': contestants,
            'Profit': profit
        })

# Create DataFrame with all data
df = pd.DataFrame(all_data)

# Add ranking information for each percentage value
df_with_ranks = pd.DataFrame()
for percentage in percentages:
    df_pct = df[df['Percentage'] == percentage].copy()
    df_pct['Rank'] = df_pct['Profit'].rank(ascending=False, method='min')
    df_with_ranks = pd.concat([df_with_ranks, df_pct])

# Calculate average rank for each box
avg_ranks = df_with_ranks.groupby('Box')['Rank'].mean().reset_index()
avg_ranks.columns = ['Box', 'Average Rank']
avg_ranks = avg_ranks.sort_values('Average Rank')

# Create a pivot table for the heatmap of ranks
rank_pivot = df_with_ranks.pivot_table(
    index='Box', 
    columns='Percentage', 
    values='Rank'
)

# Create a pivot table for profit values
profit_pivot = df_with_ranks.pivot_table(
    index='Box', 
    columns='Percentage', 
    values='Profit'
)

# Set up the figure with multiple subplots
plt.figure(figsize=(18, 20))

# 1. Line plot showing profit for each box across percentages
plt.subplot(3, 1, 1)
for box in range(1, 21):
    box_data = df_with_ranks[df_with_ranks['Box'] == box]
    
    # Color top 5 boxes differently
    if box in avg_ranks['Box'].iloc[:5].values:
        plt.plot(box_data['Percentage'], box_data['Profit'], 
                 marker='o', linewidth=2, label=f'Box {box}')
    else:
        plt.plot(box_data['Percentage'], box_data['Profit'], 
                 alpha=0.4, linewidth=1, linestyle='--')

plt.xlabel('Percentage Value', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.title('Profit for Each Box Across Different Percentage Values (1-10%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(percentages)

# Add legend for top 5 boxes
top_5_boxes = avg_ranks['Box'].iloc[:5].tolist()
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(title='Top 5 Boxes (by avg rank)')

# 2. Heatmap of box ranks for each percentage
plt.subplot(3, 1, 2)
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap', ['green', 'yellow', 'red'])

sns.heatmap(rank_pivot, cmap=custom_cmap, annot=True, fmt='.0f',
            linewidths=0.5, cbar_kws={'label': 'Rank (lower is better)'})
plt.title('Rank of Each Box Across Different Percentage Values (1-10%)', fontsize=14)
plt.xlabel('Percentage Value', fontsize=12)
plt.ylabel('Box Number', fontsize=12)

# 3. Bar chart showing average rank of each box
plt.subplot(3, 1, 3)
bars = plt.bar(avg_ranks['Box'], avg_ranks['Average Rank'])

# Highlight top 5 boxes
for i in range(5):
    bars[i].set_color('green')

plt.xlabel('Box Number', fontsize=12)
plt.ylabel('Average Rank (lower is better)', fontsize=12)
plt.title('Average Rank of Each Box Across All Percentage Values', fontsize=14)
plt.xticks(range(1, 21))
plt.grid(True, alpha=0.3, axis='y')

# Set y-axis to be inverted and limited to max 20
plt.gca().invert_yaxis()
plt.ylim(20.5, 0.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., 0.5,
             f'{20-height+1:.1f}', 
             ha='center', va='bottom', rotation=0)

plt.tight_layout()
plt.savefig('box_profit_analysis.png', dpi=300)
plt.show()

# Create tables to show the top boxes for each percentage
print("Top 5 Boxes Ranked by Profit for Each Percentage Value")
print("-" * 70)

for percentage in percentages:
    top_boxes = df_with_ranks[df_with_ranks['Percentage'] == percentage] \
        .sort_values('Profit', ascending=False) \
        .head(5)
    
    print(f"\nPercentage: {percentage}%")
    print("Rank | Box | Box Value | Contestants | Profit")
    print("-" * 50)
    
    for i, (_, row) in enumerate(top_boxes.iterrows(), 1):
        print(f"{i:4d} | {row['Box']:3d} | {row['Box Value']:9d} | {row['Contestants']:11d} | {row['Profit']:6.2f}")

# Show boxes with significant rank improvement
print("\n\nBoxes with Significant Rank Improvement as Percentage Increases:")
for box in range(1, 21):
    initial_rank = df_with_ranks[(df_with_ranks['Box'] == box) & 
                                (df_with_ranks['Percentage'] == 1)]['Rank'].values[0]
    final_rank = df_with_ranks[(df_with_ranks['Box'] == box) & 
                              (df_with_ranks['Percentage'] == 10)]['Rank'].values[0]
    improvement = initial_rank - final_rank
    
    if improvement >= 3:
        print(f"Box {box}: Improved from rank {initial_rank:.0f} at 1% to rank {final_rank:.0f} at 10% ({improvement:.0f} positions)")

# Show best box overall
best_box = avg_ranks.iloc[0]['Box']
best_avg_rank = avg_ranks.iloc[0]['Average Rank']
print(f"\nBest Box Overall: Box {best_box:.0f} with average rank {best_avg_rank:.2f}")