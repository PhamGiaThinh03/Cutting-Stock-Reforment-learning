# Re-import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("data/data.csv")

# Create a figure with multiple subplots
plt.figure(figsize=(15, 10))

# Count the number of stocks and products per batch
stock_counts = df[df["type"] == "stock"].groupby("batch_id").size()
product_counts = df[df["type"] == "product"].groupby("batch_id").size()

# Plot the number of stocks and products in each batch separately
plt.subplot(2, 2, 1)
plt.bar(stock_counts.index - 0.2, stock_counts.values, width=0.4, label="Stocks", alpha=0.7)
plt.bar(product_counts.index + 0.2, product_counts.values, width=0.4, label="Products", alpha=0.7)
plt.xlabel("Batch ID")
plt.ylabel("Count")
plt.title("Number of Stocks and Products per Batch")
plt.legend()

# Plot the size distribution (width x height) of stocks and products
plt.subplot(2, 2, 2)
plt.scatter(df[df["type"] == "stock"]["width"], df[df["type"] == "stock"]["height"], alpha=0.7, label="Stocks", marker="o")
plt.scatter(df[df["type"] == "product"]["width"], df[df["type"] == "product"]["height"], alpha=0.7, label="Products", marker="x")
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Size Distribution of Stocks and Products")
plt.legend()

# Compute the average area of stocks and products per batch
df["area"] = df["width"] * df["height"]
avg_stock_area = df[df["type"] == "stock"].groupby("batch_id")["area"].mean()
avg_product_area = df[df["type"] == "product"].groupby("batch_id")["area"].mean()

# Plot the average area of stocks and products per batch
plt.subplot(2, 2, 3)
plt.plot(avg_stock_area.index, avg_stock_area.values, marker='o', linestyle='-', label="Stocks", alpha=0.7)
plt.plot(avg_product_area.index, avg_product_area.values, marker='s', linestyle='-', label="Products", alpha=0.7)
plt.xlabel("Batch ID")
plt.ylabel("Average Area")
plt.title("Average Area of Stocks and Products per Batch")
plt.legend()

# Compute and plot the ratio of products to stocks per batch
ratio_product_stock = product_counts / stock_counts
plt.subplot(2, 2, 4)
plt.plot(ratio_product_stock.index, ratio_product_stock.values, marker='o', linestyle='-', color='r', label="Products per Stock Ratio")
plt.xlabel("Batch ID")
plt.ylabel("Ratio")
plt.title("Product count to Stock count Ratio per Batch")
plt.legend()


# Save the plot as an image file
plot_filename = "data/stock_product_analysis.png"
plt.tight_layout()
plt.savefig(plot_filename)

# Show the plots
plt.show()