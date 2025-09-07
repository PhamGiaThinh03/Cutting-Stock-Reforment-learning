import numpy as np
import pickle
import pandas as pd
from env import CuttingStockEnv
from scipy import ndimage
import random
import os

# Định nghĩa stocks và products giống file huấn luyện
stocks = [
    (100, 60),   (60, 50),   (70, 50),   (80, 60),   (90, 70),
    (100, 50),  (110, 80),  (120, 80),  (80, 90),  (120, 100),
    (60, 120), (50, 100), (110, 40), (50, 80), (90, 100)
]

products = [
    (10, 5),  (15, 10), (20, 10), (25, 15), (30, 20),
    (35, 20), (40, 30), (45, 25), (50, 30), (55, 35),
    (20, 15), (25, 10), (30, 15), (35, 20), (40, 25),
    (45, 30), (50, 35), (55, 40), (60, 45), (65, 50),
    (70, 30), (75, 40), (80, 50), (85, 55), (90, 60),
    (15, 10), (20, 15), (25, 20), (30, 25), (35, 30)
]

# Định nghĩa state_size và action_size
state_size = 10000  # Đồng bộ với file huấn luyện
action_size = 500

# Các hàm từ file huấn luyện
def get_improved_state(observation):
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    if not isinstance(observation, dict):
        return random.randint(0, state_size - 1)
    
    stocks_info = []
    if "stocks" in observation:
        stocks = observation["stocks"]
        for stock in stocks:
            if hasattr(stock, 'shape'):
                total_cells = stock.shape[0] * stock.shape[1]
                used_cells = np.sum(stock != -1)
                utilization = min(int(used_cells * 10 / total_cells), 9)
                empty_mask = (stock == -1)
                labeled, num_fragments = ndimage.label(empty_mask)
                fragmentation = min(num_fragments, 9)
                stocks_info.append((utilization, fragmentation))
    
    products_info = []
    if "products" in observation:
        products = observation["products"]
        total_product_area = 0
        max_product_area = 0
        for prod in products:
            w, h = prod["size"]
            area = w * h
            total_product_area += area
            max_product_area = max(max_product_area, area)
        normalized_area = min(int(total_product_area / 1000), 9)
        products_count = min(len(products), 9)
        max_product_size = min(int(np.sqrt(max_product_area) / 10), 9)
        products_info.extend([normalized_area, products_count, max_product_size])
    
    state_hash = 0
    for i, (util, frag) in enumerate(stocks_info[:5]):
        state_hash = state_hash * 100 + util * 10 + frag
    for info in products_info:
        state_hash = state_hash * 10 + info
    
    print(f"Debug - state_hash before modulo: {state_hash}")
    state = state_hash % state_size
    print(f"Debug - state after modulo: {state}")
    return state

def get_state(observation):
    return get_improved_state(observation)

def get_env_action(action, observation):
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    if not isinstance(observation, dict) or "products" not in observation or "stocks" not in observation:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
    
    list_prods = observation["products"]
    list_stocks = observation["stocks"]

    if not list_prods or not list_stocks:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    prod_idx = action % len(list_prods)
    prod = list_prods[prod_idx]
    prod_w, prod_h = prod["size"] if isinstance(prod, dict) else prod

    best_stock_idx = -1
    best_position = None
    best_waste = float('inf')
    best_size = None

    for w, h in [(prod_w, prod_h), (prod_h, prod_w)]:
        for stock_idx, stock in enumerate(list_stocks):
            if hasattr(stock, 'shape'):
                stock_w = np.sum(np.any(stock != -2, axis=1))
                stock_h = np.sum(np.any(stock != -2, axis=0))
                for x in range(stock_w - w + 1):
                    for y in range(stock_h - h + 1):
                        if np.all(stock[x:x + w, y:y + h] == -1):
                            waste = (stock_w - x - w) * (stock_h - y - h)
                            if waste < best_waste:
                                best_waste = waste
                                best_stock_idx = stock_idx
                                best_position = (x, y)
                                best_size = (w, h)
            else:
                stock_w, stock_h = stock
                if w <= stock_w and h <= stock_h:
                    x = random.randint(0, stock_w - w)
                    y = random.randint(0, stock_h - h)
                    waste = (stock_w - x - w) * (stock_h - y - h)
                    if waste < best_waste:
                        best_waste = waste
                        best_stock_idx = stock_idx
                        best_position = (x, y)
                        best_size = (w, h)

    if best_stock_idx != -1:
        return {"stock_idx": best_stock_idx, "size": best_size, "position": best_position}
    return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

def get_enhanced_reward(observation, info, action_taken):
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    base_reward = 0
    if isinstance(info, dict):
        filled_ratio = info.get("filled_ratio", 0.5)
        trim_loss = info.get("trim_loss", 0.2)
        base_reward += 2 * (filled_ratio - trim_loss)
    
    if isinstance(observation, dict) and "stocks" in observation:
        stocks = observation["stocks"]
        total_area_utilized = 0
        total_area_available = 0
        for stock in stocks:
            total_area_available += stock.shape[0] * stock.shape[1]
            total_area_utilized += np.sum(stock != -1)
        if total_area_available > 0:
            utilization_ratio = total_area_utilized / total_area_available
            base_reward += 5 * (utilization_ratio ** 2)
    
    if isinstance(action_taken, dict) and "size" in action_taken:
        width, height = action_taken["size"]
        piece_area = width * height
        base_reward += 0.2 * np.sqrt(piece_area) if piece_area > 0 else 0
    
    if isinstance(observation, dict) and "products" in observation:
        remaining_products = sum(p["quantity"] for p in observation["products"])
        if remaining_products == 0:
            base_reward += 200
        else:
            base_reward -= 10 * remaining_products
    
    if action_taken is None or (isinstance(action_taken, dict) and action_taken.get("size", (0, 0)) == (0, 0)):
        base_reward -= 20
    
    return base_reward

def test_q_learning(env, q_table, num_tests=100, render=True, max_steps=300, epsilon=0.1):
    results = []
    
    print("\nStarting Q-learning testing...")
    for test_idx in range(num_tests):
        observation, info = env.reset(seed=42 + test_idx)
        state = get_state(observation)
        total_reward = 0
        steps = 0
        done = False
        used_stocks = set()
        
        while not done and steps < max_steps:
            if render:
                env.render()
            
            if state >= state_size:
                state = state % state_size
                print(f"Warning: State {state} was out of bounds, adjusted to {state % state_size}")
            
            if random.random() < epsilon:
                action_idx = random.randint(0, action_size - 1)
            else:
                action_idx = np.argmax(q_table[state])
            
            env_action = get_env_action(action_idx, observation)
            print(f"Test {test_idx + 1}, Step {steps}: Action={env_action}")
            
            try:
                step_result = env.step(env_action)
                observation, reward, done, _, info = step_result
                total_reward += get_enhanced_reward(observation, info, env_action)
                state = get_state(observation)
                steps += 1
                
                stock_idx = env_action["stock_idx"]
                if env_action["size"] != (0, 0):
                    used_stocks.add(stock_idx)
            except Exception as e:
                print(f"Error in test step: {e}")
                done = True
        
        results.append({
            "Test": test_idx + 1,
            "Total Reward": total_reward,
            "Steps": steps,
            "Used Stocks": len(used_stocks),
            "Trim Loss": info.get("trim_loss", 0),
            "Remaining Products": info.get("remaining_products", 0)
        })
        
        print(f"Test {test_idx + 1}: Reward={total_reward:.4f}, Steps={steps}, "
              f"Used Stocks={len(used_stocks)}, Trim Loss={info.get('trim_loss', 0):.4f}, "
              f"Remaining Products={info.get('remaining_products', 0)}")
    
    avg_results = {
        "Avg Reward": np.mean([r["Total Reward"] for r in results]),
        "Avg Steps": np.mean([r["Steps"] for r in results]),
        "Avg Used Stocks": np.mean([r["Used Stocks"] for r in results]),
        "Avg Trim Loss": np.mean([r["Trim Loss"] for r in results]),
        "Avg Remaining Products": np.mean([r["Remaining Products"] for r in results])
    }
    
    print("\nTest Summary:")
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")
    
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/q_learning_test_results.csv", index=False)
    print("Test results saved to results/q_learning_test_results.csv")
    
    return results, avg_results

def main():
    global env
    env = CuttingStockEnv(
        render_mode="human",
        max_w=120,
        max_h=120,
        seed=42,
        stock_list=stocks,
        product_list=products
    )
    
    if not hasattr(env, 'get_state'):
        def get_env_state(self):
            return self._get_obs()
        setattr(env.__class__, 'get_state', get_env_state)
    
    q_table_path = "results/q_table.pkl"
    if not os.path.exists(q_table_path):
        print(f"Error: Q-table file {q_table_path} not found. Please train first.")
        return
    
    with open(q_table_path, "rb") as f:
        q_table = pickle.load(f)
    print(f"Loaded Q-table from {q_table_path}")
    
    test_results, avg_test_results = test_q_learning(
        env=env,
        q_table=q_table,
        num_tests=100,
        render=True,
        max_steps=300,
        epsilon=0.1
    )
    
    env.close()

if __name__ == "__main__":
    main()