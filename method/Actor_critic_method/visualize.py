import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from env.cutting_stock import CuttingStockEnv
import imageio
import os

# Hyperparameters (giữ nguyên từ code huấn luyện)
MAX_W = 100
MAX_H = 100
MAX_PRODUCTS = 100

# Đọc dữ liệu từ file CSV (giống code huấn luyện)
def load_data(file_path):
    df = pd.read_csv(file_path)
    batches = df['batch_id'].unique()
    data = {}
    for batch in batches:
        batch_df = df[df['batch_id'] == batch]
        stock_list = [(row['width'], row['height']) for _, row in batch_df[batch_df['type'] == 'stock'].iterrows()]
        product_list = [(row['width'], row['height']) for _, row in batch_df[batch_df['type'] == 'product'].iterrows()]
        data[batch] = {'stocks': stock_list, 'products': product_list}
    return data

# Mô hình Actor-Critic (giống code huấn luyện)
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# Chuyển đổi state (giống code huấn luyện)
def process_state(obs, num_stocks):
    stock_info = []
    for stock in obs['stocks']:
        stock_width = np.sum(np.any(stock != -2, axis=1))
        stock_height = np.sum(np.any(stock != -2, axis=0))
        free_space = np.sum(stock == -1)
        stock_info.extend([stock_width, stock_height, free_space])
    stock_flat = np.array(stock_info, dtype=np.float32)
    
    product_flat = np.concatenate([[p['size'][0], p['size'][1], p['quantity']] 
                                   for p in obs['products']])
    
    stock_padding = np.zeros(3 * num_stocks - len(stock_flat), dtype=np.float32)
    product_padding = np.zeros(3 * MAX_PRODUCTS - len(product_flat), dtype=np.float32)
    state = np.concatenate([stock_flat, stock_padding, product_flat, product_padding])
    return torch.FloatTensor(state)

# Tìm vị trí hợp lệ trên stock (giữ nguyên)
def find_valid_position(stock, size):
    width, height = size
    stock_width, stock_height = stock.shape
    for x in range(stock_width - width + 1):
        for y in range(stock_height - height + 1):
            if np.all(stock[x:x+width, y:y+height] == -1):
                return np.array([x, y])
    return None

# Chọn hành động (không dùng epsilon vì chỉ visualize)
def select_action(model, state, env):
    probs, value = model(state)
    m = Categorical(probs)
    action_idx = m.sample()
    stock_idx = action_idx % env.action_space['stock_idx'].n
    
    for product_idx, product in enumerate(env._products):
        if product['quantity'] > 0:
            size = product['size']
            break
    else:
        size = np.array([1, 1])
    
    stock = env._stocks[stock_idx]
    position = find_valid_position(stock, size)
    if position is None:
        for i in range(env.action_space['stock_idx'].n):
            if i != stock_idx:
                stock = env._stocks[i]
                position = find_valid_position(stock, size)
                if position is not None:
                    stock_idx = i
                    break
        if position is None:
            position = np.array([0, 0])
    
    action = {'stock_idx': stock_idx, 'size': size, 'position': position}
    return action

# Visualize, lưu GIF và vẽ biểu đồ đánh giá
def visualize_and_save_gif(file_path, model_path, batch_id=2, max_steps=400):
    # Đọc dữ liệu
    data = load_data(file_path)
    batch_data = data[batch_id]
    print(f"Stocks: {batch_data['stocks']}")
    print(f"Products: {batch_data['products']}")
    
    # Khởi tạo môi trường với render_mode='rgb_array'
    env = CuttingStockEnv(
        render_mode='rgb_array',  # Để lấy frame hình ảnh
        max_w=MAX_W,
        max_h=MAX_H,
        stock_list=batch_data['stocks'],
        product_list=batch_data['products'],
        seed=42
    )
    
    # Khởi tạo mô hình và tải trọng số
    state_size = 3 * env.num_stocks + 3 * MAX_PRODUCTS
    action_size = env.action_space['stock_idx'].n
    model = ActorCritic(state_size, action_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Chuyển sang chế độ đánh giá (không huấn luyện)
    
    # Reset môi trường
    state, _ = env.reset()
    state = process_state(state, env.num_stocks)
    done = False
    step = 0
    frames = []  # Danh sách lưu các frame hình ảnh
    total_reward = 0
    rewards = []  # Lưu reward qua các bước
    filled_ratios = []  # Lưu filled_ratio qua các bước
    trim_losses = []  # Lưu trim_loss qua các bước
    
    # Mô phỏng quá trình cắt
    while not done and step < max_steps:
        # Chọn hành động
        action = select_action(model, state, env)
        next_state, reward, done, _, info = env.step(action)
        
        # Tính reward (giống code huấn luyện)
        if reward > 0:
            reward = 1.0 - info['trim_loss']
        elif info['filled_ratio'] > 0:
            reward = 0.5 - info['trim_loss']
        else:
            reward = -0.1
        
        total_reward += reward
        rewards.append(reward)
        filled_ratios.append(info['filled_ratio'])
        trim_losses.append(info['trim_loss'])
        
        # Lấy frame từ môi trường
        frame = env.render()
        frames.append(frame)
        
        # In thông tin
        print(f"Step {step}: Action: Stock {action['stock_idx']}, Size {action['size']}, Position {action['position']}, "
              f"Reward: {reward:.2f}, Filled Ratio: {info['filled_ratio']:.4f}, Trim Loss: {info['trim_loss']:.4f}")
        
        # Cập nhật state
        state = process_state(next_state, env.num_stocks)
        step += 1
    
    # In kết quả cuối cùng
    print(f"Visualization completed: Total Reward: {total_reward:.2f}, Trim Loss: {info['trim_loss']:.4f}, "
          f"Filled Ratio: {info['filled_ratio']:.4f}")
    
    # Lưu GIF
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f'cutting_batch_{batch_id}_visualization.gif')
    imageio.mimsave(gif_path, frames, fps=10)  # 10 FPS
    print(f"GIF saved to {gif_path}")
    
    # Vẽ biểu đồ đánh giá
    steps = list(range(len(rewards)))
    
    plt.figure(figsize=(15, 5))
    
    # Biểu đồ Reward
    plt.subplot(1, 3, 1)
    plt.plot(steps, rewards, label='Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward over Steps')
    plt.legend()
    
    # Biểu đồ Filled Ratio
    plt.subplot(1, 3, 2)
    plt.plot(steps, filled_ratios, label='Filled Ratio', color='green')
    plt.xlabel('Step')
    plt.ylabel('Filled Ratio')
    plt.title('Filled Ratio over Steps')
    plt.legend()
    
    # Biểu đồ Trim Loss
    plt.subplot(1, 3, 3)
    plt.plot(steps, trim_losses, label='Trim Loss', color='red')
    plt.xlabel('Step')
    plt.ylabel('Trim Loss')
    plt.title('Trim Loss over Steps')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'performance_batch_{batch_id}.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Performance plot saved to {plot_path}")
    
    # Đóng môi trường
    env.close()

if __name__ == "__main__":
    file_path = "data/data.csv"  # Đường dẫn đến file dữ liệu
    model_path = "actor_critic_batch_1.pth"  # Đường dẫn đến file mô hình .pth
    visualize_and_save_gif(file_path, model_path, batch_id=2)