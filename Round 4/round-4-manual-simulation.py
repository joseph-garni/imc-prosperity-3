import pandas as pd
import matplotlib.pyplot as plt

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
    return (box_value * 10) / (percentage + num_contestants)

def calculate_and_display_profits(box_percentages):
    """
    Calculate profits for each box based on custom percentages and display results
    
    Parameters:
    box_percentages (dict): Dictionary with box numbers as keys and percentage values as values
    """
    results = []
    
    # Calculate profit for each box
    for box_num, (box_value, contestants) in sample_data.items():
        # Use the custom percentage for this box, default to 5% if not specified
        percentage = box_percentages.get(box_num, 5)
        profit = calculate_profit(box_value, contestants, percentage)
        
        results.append({
            'Box': box_num,
            'Box Value': box_value,
            'Contestants': contestants,
            'Percentage': percentage,
            'Profit': profit
        })
    
    # Create DataFrame and sort by profit
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('Profit', ascending=False)
    
    # Add rank column
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    # Display results
    print("\nBox Profit Results (sorted by profit):\n")
    print(df_sorted[['Box', 'Box Value', 'Contestants', 'Percentage', 'Profit', 'Rank']].to_string(index=False))
    
    # Display top 5 boxes
    print("\nTop 5 Most Profitable Boxes:")
    for i, row in df_sorted.head(5).iterrows():
        print(f"#{row['Rank']}: Box {row['Box']} - Profit: {row['Profit']:.2f} (using {row['Percentage']}%)")
    
    # Optional: Plot the results
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_sorted['Box'], df_sorted['Profit'])
    
    # Color the top 5 bars
    for i in range(5):
        bars[i].set_color('green')
    
    plt.xlabel('Box Number')
    plt.ylabel('Profit')
    plt.title('Box Profits with Custom Percentages')
    plt.xticks(range(1, 21))
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    return df_sorted

def main():
    print("Box Profit Calculator with Custom Percentages")
    print("---------------------------------------------")
    print("Enter custom percentage values for each box (or leave blank for default 5%)")
    print("Type 'done' when finished or 'all' to set same percentage for all boxes")
    
    box_percentages = {}
    
    # Option to set all percentages at once
    all_input = input("\nSet same percentage for all boxes? (y/n): ")
    if all_input.lower() == 'y':
        try:
            all_pct = float(input("Enter percentage for all boxes: "))
            for box_num in range(1, 21):
                box_percentages[box_num] = all_pct
            
            # Calculate and display results
            result_df = calculate_and_display_profits(box_percentages)
            return
        except ValueError:
            print("Invalid input. Using default 5% for all boxes.")
    
    # Individual box percentages
    while True:
        try:
            box_input = input("\nEnter box number (1-20) or 'done': ")
            
            if box_input.lower() == 'done':
                break
            
            box_num = int(box_input)
            if box_num < 1 or box_num > 20:
                print("Box number must be between 1 and 20")
                continue
            
            pct_input = input(f"Enter percentage for Box {box_num}: ")
            box_percentages[box_num] = float(pct_input)
            
            # Show current settings
            print("\nCurrent percentage settings:")
            for box, pct in box_percentages.items():
                print(f"Box {box}: {pct}%")
            
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    
    # Calculate and display results
    result_df = calculate_and_display_profits(box_percentages)

if __name__ == "__main__":
    main()