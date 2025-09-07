import time
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import cv2
import numpy as np
from policy.FirstFit_Policy import first_fit_policy
from policy.BestFit_Policy import best_fit_policy
from policy.Combination_Policy import combination_policy
from env import CuttingStockEnv
from env.renderer import render_frame  # Import render_frame từ module renderer

# Create results directories
os.makedirs("results/gifs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Read data from CSV
df = pd.read_csv("data/data.csv")
batch_ids = df["batch_id"].unique()
results = []

def add_text_to_frame(frame, text):
    # Convert frame to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Add text to frame
    cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    # Convert back to RGB
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

for batch_id in batch_ids:
    print(f"-----batch_id: {batch_id}-----")
    df_batch = df[df["batch_id"] == batch_id]
    all_policy_frames = []
    
    stock_list = df_batch[df_batch["type"] == "stock"][["width", "height"]].to_records(index=False).tolist()
    product_list = df_batch[df_batch["type"] == "product"][["width", "height"]].to_records(index=False).tolist()

    for policy_name, policy_fn in [("First-Fit", first_fit_policy), 
                                 ("Best-Fit", best_fit_policy), 
                                 ("Combination", combination_policy)]:
        print(f"Policy: {policy_name}")
        
        env = CuttingStockEnv(
            render_mode="rgb_array",
            max_w=200,
            max_h=200,
            seed=42,
            stock_list=stock_list,
            product_list=product_list,
        )

        obs, info = env.reset()
        frames = []
        
        # Sửa lỗi: Sử dụng render_frame thay vì _render_frame
        initial_frame = render_frame(env)
        labeled_frame = add_text_to_frame(initial_frame, policy_name)
        frames.append(labeled_frame)

        start_time = time.time()
        total_reward, steps = 0, 0
        done = False

        while not done:
            action = policy_fn(obs, info)
            obs, reward, done, _, info = env.step(action)
            # Sửa lỗi: Sử dụng render_frame thay vì _render_frame
            frame = render_frame(env)
            labeled_frame = add_text_to_frame(frame, policy_name)
            frames.append(labeled_frame)
            total_reward += reward
            steps += 1

        end_time = time.time()
        runtime = end_time - start_time
        
        # Add frames to combined list
        all_policy_frames.extend(frames)
        
        # Add separator frames
        blank_frame = np.zeros_like(frames[0])
        separator_frame = add_text_to_frame(blank_frame, f"Next: {policy_name}")
        for _ in range(10):  # Longer pause between policies
            all_policy_frames.append(separator_frame)

        # Calculate metrics
        remaining_stocks = sum(1 for stock in env._stocks if np.all(stock[stock != -2] == -1))
        used_stocks = len(env._stocks) - remaining_stocks
        total_trim_loss = sum((stock == -1).sum() for stock in env._stocks if np.any(stock[stock != -2] != -1))
        
        used_stock_areas = [np.sum(stock != -2) for stock in env._stocks if np.any(stock[stock != -2] != -1)]
        avg_used_stock_area = sum(used_stock_areas) / len(used_stock_areas) if used_stock_areas else 0

        results.append({
            "batch_id": batch_id,
            "policy": policy_name,
            "steps": steps,
            "runtime": runtime,
            "total_trim_loss": total_trim_loss,
            "remaining_stocks": remaining_stocks,
            "used_stocks": used_stocks,
            "avg_used_stock_area": avg_used_stock_area
        })

        env.close()

    # Save combined GIF for this batch
    gif_path = f"results/gifs/batch_{batch_id}_all_policies.gif"
    imageio.mimsave(gif_path, all_policy_frames, fps=5)
    print(f"Saved combined visualization to {gif_path}")

# Save results to CSV
df_results = pd.DataFrame(results)
df_results.to_csv("data/policy_comparison_results.csv", index=False)
print("Results saved to data/policy_comparison_results.csv")