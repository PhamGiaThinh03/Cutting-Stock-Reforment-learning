#lưu file gif chạy file main.py 
import imageio
import os
from policy import FirstFit_Policy, BestFit_Policy, Combination_Policy
from env.cutting_stock import CuttingStockEnv

# Danh sách stocks (width, height) - Tấm nguyên liệu có kích thước nhỏ, tối đa 200x200
stocks = [
    (50, 50),   (60, 40),   (70, 50),   (80, 60),   (90, 70),
    (100, 50),  (110, 60),  (120, 80),  (130, 90),  (140, 100),
    (150, 120), (160, 130), (170, 140), (180, 150), (200, 200)
]

# Danh sách products (width, height) - Sản phẩm có kích thước nhỏ, phù hợp với stocks
products = [
    (10, 5),  (15, 10), (20, 10), (25, 15), (30, 20),
    (35, 20), (40, 30), (45, 25), (50, 30), (55, 35),
    (20, 15), (25, 10), (30, 15), (35, 20), (40, 25),
    (45, 30), (50, 35), (55, 40), (60, 45), (65, 50),
    (35, 20), (40, 30), (45, 25), (50, 30), (55, 35),
    (20, 15), (25, 10), (30, 15), (35, 20), (40, 25),
    (45, 30), (50, 35), (55, 40), (60, 45), (65, 50),
    (70, 30), (75, 40), (80, 50), (85, 55), (90, 60),
    (35, 20), (40, 30), (45, 25), (50, 30), (55, 35),
    (20, 15), (25, 10), (30, 15), (35, 20), (40, 25),
    (45, 30), (50, 35), (55, 40), (60, 45), (65, 50),
    (70, 30), (75, 40), (80, 50), (85, 55), (90, 60),
    (15, 10), (20, 15), (25, 20), (30, 25), (35, 30)
]


env = CuttingStockEnv(
    render_mode="human",   # hoặc None nếu không cần hiển thị
    max_w=200,             # Giá trị max_w, max_h nên lớn hơn hoặc bằng kích thước của stocks
    max_h=200,
    seed=42,
    stock_list=stocks,
    product_list=products,
) 

if __name__ == "__main__":
    done = False
    observation, info = env.reset()
    while not done:
        # action = FirstFit_Policy.first_fit_policy(observation, info)
        action = BestFit_Policy.best_fit_policy(observation, info)
        # action = Combination_Policy.combination_policy(observation, info)
        observation, reward, done, truncated, info = env.step(action)
    env.close()
    

