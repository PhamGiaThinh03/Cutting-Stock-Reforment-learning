import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from env.cutting_stock import CuttingStockEnv

# Hyperparameters
GAMMA = 0.99
LR = 0.001
MAX_EPISODES = 1000  # Tăng số episode
MAX_STEPS = 200  # Tăng số bước
MAX_W = 150
MAX_H = 150
MAX_PRODUCTS = 100
EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Đọc dữ liệu từ file CSV
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

# Mô hình Actor-Critic
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

# Chuyển đổi state
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

# Tìm vị trí hợp lệ trên stock
def find_valid_position(stock, size):
    width, height = size
    stock_width, stock_height = stock.shape  # Dùng kích thước thực
    for x in range(stock_width - width + 1):
        for y in range(stock_height - height + 1):
            if np.all(stock[x:x+width, y:y+height] == -1):
                return np.array([x, y])
    return None

# Chọn hành động hợp lệ
def select_action(model, state, env, epsilon):
    probs, value = model(state)
    m = Categorical(probs)
    
    if np.random.rand() < epsilon:
        stock_idx = np.random.randint(env.action_space['stock_idx'].n)
    else:
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
    print(f"Action: Stock {stock_idx}, Size {size}, Position {position}")
    return action, m.log_prob(torch.tensor(stock_idx)), value

# Huấn luyện
def train(file_path, batch_id=1):
    data = load_data(file_path)
    batch_data = data[batch_id]
    print(f"Stocks: {batch_data['stocks']}")
    print(f"Products: {batch_data['products']}")
    
    env = CuttingStockEnv(
        render_mode=None,
        max_w=MAX_W,
        max_h=MAX_H,
        stock_list=batch_data['stocks'],
        product_list=batch_data['products'],
        seed=42
    )
    
    state_size = 3 * env.num_stocks + 3 * MAX_PRODUCTS
    action_size = env.action_space['stock_idx'].n
    model = ActorCritic(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    episode_rewards = []
    policy_losses = []
    value_losses = []
    epsilon = EPSILON_START
    
    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        state = process_state(state, env.num_stocks)
        log_probs = []
        values = []
        rewards = []
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            action, log_prob, value = select_action(model, state, env, epsilon)
            next_state, reward, done, _, info = env.step(action)
            next_state = process_state(next_state, env.num_stocks)
            
            if reward > 0:
                reward = 1.0 - info['trim_loss']
            elif info['filled_ratio'] > 0:
                reward = 0.5 - info['trim_loss']
            else:
                reward = -0.1
            
            print(f"Step {step}: Reward: {reward:.2f}, Filled Ratio: {info['filled_ratio']:.4f}, Trim Loss: {info['trim_loss']:.4f}")
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            state = next_state
            step += 1
        
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        policy_loss = []
        value_loss = []
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward()
        optimizer.step()
        
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        policy_losses.append(torch.stack(policy_loss).sum().item())
        value_losses.append(torch.stack(value_loss).sum().item())
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Trim Loss: {info['trim_loss']:.4f}, Filled Ratio: {info['filled_ratio']:.4f}, Epsilon: {epsilon:.4f}")
    
    torch.save(model.state_dict(), f"actor_critic_batch_{batch_id}.pth")
    print(f"Model saved as actor_critic_batch_{batch_id}.pth")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Episodes')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(policy_losses, label='Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Policy Loss over Episodes')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(value_losses, label='Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Value Loss over Episodes')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"training_results_batch_{batch_id}.png")
    plt.show()

if __name__ == "__main__":
    file_path = "data/data.csv"
    train(file_path, batch_id=1)