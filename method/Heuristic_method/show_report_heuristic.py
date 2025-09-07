import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu từ file CSV
csv_filename = "data/policy_comparison_results.csv"
df_results = pd.read_csv(csv_filename)

# Tạo biểu đồ so sánh
plt.figure(figsize=(18, 10))

# Định nghĩa kiểu đường và màu sắc
linestyles = {"First-Fit": "-", "Best-Fit": "--", "Combination": "-."}
colors = {"First-Fit": "r", "Best-Fit": "g", "Combination": "b"}

# Thêm jitter để tránh trùng lặp điểm dữ liệu
jitter_scale = 0.1

# Vẽ biểu đồ thời gian chạy
plt.subplot(2, 3, 1)
for policy in df_results["policy"].unique():
    jitter = np.random.normal(0, jitter_scale, len(df_results[df_results["policy"] == policy]["batch_id"]))
    plt.plot(df_results[df_results["policy"] == policy]["batch_id"] + jitter, 
             df_results[df_results["policy"] == policy]["runtime"], 
             label=policy, marker='o', linestyle=linestyles[policy], color=colors[policy], alpha=0.7)
plt.xlabel("Batch ID")
plt.ylabel("Runtime (s)")
plt.title("Runtime Comparison")
plt.legend()

# Vẽ biểu đồ tổng diện tích phần thừa của các stock đã sử dụng
plt.subplot(2, 3, 2)
for policy in df_results["policy"].unique():
    jitter = np.random.normal(0, jitter_scale, len(df_results[df_results["policy"] == policy]["batch_id"]))
    plt.plot(df_results[df_results["policy"] == policy]["batch_id"] + jitter, 
             df_results[df_results["policy"] == policy]["total_trim_loss"], 
             label=policy, marker='o', linestyle=linestyles[policy], color=colors[policy], alpha=0.7)
plt.xlabel("Batch ID")
plt.ylabel("Total Trim Loss")
plt.title("Total Trim Loss Comparison")
plt.legend()

# Vẽ biểu đồ số lượng stock còn lại trong kho mà không sử dụng
plt.subplot(2, 3, 3)
for policy in df_results["policy"].unique():
    jitter = np.random.normal(0, jitter_scale, len(df_results[df_results["policy"] == policy]["batch_id"]))
    plt.plot(df_results[df_results["policy"] == policy]["batch_id"] + jitter, 
             df_results[df_results["policy"] == policy]["remaining_stocks"], 
             label=policy, marker='o', linestyle=linestyles[policy], color=colors[policy], alpha=0.7)
plt.xlabel("Batch ID")
plt.ylabel("Remaining Stocks")
plt.title("Remaining Stocks Comparison")
plt.legend()

# Vẽ biểu đồ số lượng stock đã được sử dụng
plt.subplot(2, 3, 4)
for policy in df_results["policy"].unique():
    jitter = np.random.normal(0, jitter_scale, len(df_results[df_results["policy"] == policy]["batch_id"]))
    plt.plot(df_results[df_results["policy"] == policy]["batch_id"] + jitter, 
             df_results[df_results["policy"] == policy]["used_stocks"], 
             label=policy, marker='o', linestyle=linestyles[policy], color=colors[policy], alpha=0.7)
plt.xlabel("Batch ID")
plt.ylabel("Used Stocks")
plt.title("Used Stocks Comparison")
plt.legend()

# Vẽ biểu đồ diện tích trung bình của các stock đã được sử dụng
plt.subplot(2, 3, 5)
for policy in df_results["policy"].unique():
    jitter = np.random.normal(0, jitter_scale, len(df_results[df_results["policy"] == policy]["batch_id"]))
    plt.plot(df_results[df_results["policy"] == policy]["batch_id"] + jitter, 
             df_results[df_results["policy"] == policy]["avg_used_stock_area"], 
             label=policy, marker='o', linestyle=linestyles[policy], color=colors[policy], alpha=0.7)
plt.xlabel("Batch ID")
plt.ylabel("Avg Used Stock Area")
plt.title("Avg Used Stock Area Comparison")
plt.legend()


# Save the plot as an image file
plot_filename = "results/comparison.png"
plt.tight_layout()
plt.savefig(plot_filename)

# Show the plots
plt.show()