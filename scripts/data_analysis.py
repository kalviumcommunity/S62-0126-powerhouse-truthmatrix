# data_analysis.py
# Milestone 4.13 - Basic Python Script for Data Analysis

print("Starting Data Analysis Script...\n")

# Sample data (daily sales)
sales = [1200, 1500, 1100, 1800, 1700]

# Basic calculations
total_sales = sum(sales)
average_sales = total_sales / len(sales)
max_sales = max(sales)
min_sales = min(sales)

# Output results
print("Sales Data:", sales)
print("Total Sales:", total_sales)
print("Average Sales:", average_sales)
print("Highest Sale:", max_sales)
print("Lowest Sale:", min_sales)

print("\nScript executed successfully!")