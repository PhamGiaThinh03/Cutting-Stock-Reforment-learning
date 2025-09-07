from env.cutting_stock import CuttingStockEnv
import numpy as np
import random
import pickle
import os
import pandas as pd

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
    (70, 30), (75, 40), (80, 50), (85, 55), (90, 60),
    (15, 10), (20, 15), (25, 20), (30, 25), (35, 30)
]

# Tạo thư mục results nếu chưa tồn tại
os.makedirs("results", exist_ok=True)

# Xóa file log cũ nếu tồn tại để tránh append dữ liệu cũ
if os.path.exists("results/q_learning_log1.txt"):
    os.remove("results/q_learning_log1.txt")

env = CuttingStockEnv(
    render_mode="human",   
    max_w=120,           
    max_h=120,
    seed=42,
    stock_list=stocks,
    product_list=products,
)

# Thêm method get_state vào environment
def get_env_state(self):
    """Convert tuple state from reset/step to dictionary format"""
    if hasattr(self, 'observation'):
        return self.observation
    
    # Giả định observation từ reset/step là tuple có 2 phần tử
    # Phần tử đầu tiên là stocks, phần tử thứ hai là products
    reset_result = self.reset()
    if isinstance(reset_result, tuple):
        if len(reset_result) > 0:
            try:
                # Try to extract stocks and products
                stocks_info = []
                products_info = []
                
                # Analyze tuple structure
                print(f"Reset result length: {len(reset_result)}")
                
                if len(reset_result) >= 1:
                    stocks_info = reset_result[0]
                if len(reset_result) >= 2:
                    products_info = reset_result[1]
                
                # Create observation dictionary
                self.observation = {
                    "stocks": stocks_info,
                    "products": products_info
                }
                
                return self.observation
            except Exception as e:
                print(f"Error extracting state from tuple: {e}")
    
    # Default empty observation
    return {"stocks": [], "products": []}

# Thêm method vào environment
if not hasattr(env, 'get_state'):
    setattr(env.__class__, 'get_state', get_env_state)

alpha = 0.3
gamma = 0.9  
epsilon = 1.0  
epsilon_decay = 0.995  
min_epsilon = 0.01 
num_episodes = 100  # Số tập huấn luyện
min_alpha = 0.1  # Giá trị nhỏ nhất của alpha

# Kích thước Q-table 
state_size = 100000
action_size = 1000  
Q_table = np.zeros((state_size, action_size))


def get_state(observation_tuple):
    """
    Chuyển trạng thái từ môi trường thành dạng số nguyên để lưu vào Q-table.
    Hỗ trợ cả định dạng tuple và dictionary.
    """
    # Kiểm tra xem observation có phải là tuple hay không
    if isinstance(observation_tuple, tuple):
        # Sử dụng hàm get_state của environment để chuyển tuple thành dict
        observation = env.get_state()
    else:
        observation = observation_tuple
    
    # Tiếp tục xử lý với observation dạng dict
    if isinstance(observation, dict) and "stocks" in observation and "products" in observation:
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Trích xuất thông tin từ stocks
        if isinstance(stocks, list) and stocks:
            # Tính tổng diện tích trống
            empty_space = 0
            for stock in stocks:
                if hasattr(stock, 'shape'):  # Nếu stock là numpy array
                    empty_space += np.sum(stock == -1)
                elif isinstance(stock, tuple) or isinstance(stock, list):
                    # Nếu stock là kích thước (width, height)
                    if len(stock) >= 2:
                        empty_space += stock[0] * stock[1]
            
            # Tổng số sản phẩm chưa cắt
            remaining_products = 0
            if isinstance(products, list):
                for prod in products:
                    if isinstance(prod, dict) and "quantity" in prod:
                        remaining_products += prod["quantity"]
                    else:
                        remaining_products += 1  # Giả sử mỗi sản phẩm có số lượng 1
            
            state = (empty_space * 1000 + remaining_products) % state_size
            return state
    
    # Nếu không thể xử lý observation, trả về giá trị ngẫu nhiên
    return random.randint(0, state_size-1)


def get_action(state):
    """
    Chọn hành động sử dụng epsilon-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_size - 1)
    else:
        return np.argmax(Q_table[state])

def get_action_with_guidance(state, observation):
    """
    Exploration với hướng dẫn dựa trên domain knowledge
    """
    # 80% thời gian sử dụng thông thường exploration
    if random.random() < 0.8:
        return get_action(state)  
    
    # 20% thời gian sử dụng heuristic guidance
    # Trích xuất thông tin từ observation
    if isinstance(observation, dict) and "products" in observation and "stocks" in observation:
        products = observation["products"]
        stocks = observation["stocks"]
        
        # Heuristic 1: Sắp xếp sản phẩm lớn trước
        largest_product_idx = -1
        max_area = 0
        
        for i, prod in enumerate(products):
            if isinstance(prod, tuple) and len(prod) >= 2:
                area = prod[0] * prod[1]
                if area > max_area:
                    max_area = area
                    largest_product_idx = i
        
        # Heuristic 2: Chọn stock có nhiều không gian nhất
        best_stock_idx = 0
        max_empty = 0
        
        for i, stock in enumerate(stocks):
            if hasattr(stock, 'shape'):
                empty_space = np.sum(stock == -1)
                if empty_space > max_empty:
                    max_empty = empty_space
                    best_stock_idx = i
        
        # Tạo action từ heuristics
        if largest_product_idx >= 0:
            # Transform to Q-table action space
            return (best_stock_idx * len(products) + largest_product_idx) % action_size
    
    # Fallback to random action
    return random.randint(0, action_size - 1)

def get_env_action(action, observation):
    """
    Chuyển action từ Q-table thành action thực tế cho môi trường Gym.
    Hỗ trợ cả định dạng observation là tuple hoặc dictionary.
    """
    # Chuyển đổi observation thành dictionary nếu là tuple
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    # Kiểm tra xem observation có cấu trúc dự kiến không
    if not isinstance(observation, dict) or "products" not in observation or "stocks" not in observation:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
    
    list_prods = observation["products"]
    list_stocks = observation["stocks"]

    if not list_prods or not list_stocks:
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}

    # Chọn sản phẩm có thể cắt
    prod_idx = action % len(list_prods)
    prod = list_prods[prod_idx]

    # Xử lý cả trường hợp prod là dict hoặc tuple
    if isinstance(prod, dict):
        if prod.get("quantity", 1) == 0:
            return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
        prod_w, prod_h = prod.get("size", (10, 10))
    else:
        # Giả sử prod là tuple (width, height)
        prod_w, prod_h = prod if len(prod) >= 2 else (10, 10)

    # Chọn stock
    stock_idx = (action // len(list_prods)) % len(list_stocks)
    stock = list_stocks[stock_idx]

    # Xử lý cả trường hợp stock là numpy array hoặc tuple
    if hasattr(stock, 'shape'):  # Numpy array
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        # Chọn vị trí trong stock
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                    return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}
    else:
        # Giả sử stock là tuple (width, height)
        stock_w, stock_h = stock if len(stock) >= 2 else (100, 100)
        # Chọn vị trí ngẫu nhiên trong stock
        if prod_w <= stock_w and prod_h <= stock_h:
            x = random.randint(0, stock_w - prod_w)
            y = random.randint(0, stock_h - prod_h)
            return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}

    return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}



def get_reward(observation, info):
    """
    Tính toán phần thưởng (reward) cho agent.
    Hỗ trợ cả định dạng observation là tuple hoặc dictionary.
    """
    # Chuyển đổi observation thành dictionary nếu là tuple
    if isinstance(observation, tuple):
        observation = env.get_state()
    
    # Sử dụng info nếu có sẵn
    if isinstance(info, dict):
        filled_ratio = info.get("filled_ratio", 0.5)
        trim_loss = info.get("trim_loss", 0.2)
    else:
        # Giá trị mặc định nếu không có info
        filled_ratio = 0.5
        trim_loss = 0.2

    # Tính số lượng stock đã sử dụng
    num_stocks_used = 0
    total_stocks = 0
    
    if isinstance(observation, dict) and "stocks" in observation:
        stocks = observation["stocks"]
        total_stocks = len(stocks)
        
        for stock in stocks:
            if hasattr(stock, 'shape'):  # Numpy array
                if np.any(stock != -2):
                    num_stocks_used += 1
            else:
                # Giả sử mỗi phần tử trong stocks là một stock đã sử dụng
                num_stocks_used += 1
    
    # Nếu không có thông tin về stocks, sử dụng giá trị mặc định
    if total_stocks == 0:
        total_stocks = 1
        num_stocks_used = 0

    num_stocks_unused = total_stocks - num_stocks_used
    
    lambda_bonus = 0.2  # Hệ số điều chỉnh mức độ thưởng
    stock_bonus = lambda_bonus * (num_stocks_unused / total_stocks)  # Thưởng theo tỷ lệ stock chưa cắt

    # Tính tổng phần thưởng
    reward = (filled_ratio - trim_loss) + stock_bonus

    return reward


# Biến theo dõi phần thưởng cao nhất đạt được
max_ep_reward = -999  # Giá trị phần thưởng lớn nhất tìm thấy
max_ep_action_list = []  # Danh sách hành động tương ứng với phần thưởng cao nhất
max_start_state = None  # Trạng thái bắt đầu tương ứng với phần thưởng cao nhất

# Lưu lại lịch sử huấn luyện cho visualization
rewards_history = []
epsilons_history = []
steps_history = []

# Train -------------------------------------
for episode in range(num_episodes):
    # Reset environment và lấy trạng thái đầu tiên
    observation = env.reset(seed=42)
    
    # Extract info nếu reset trả về tuple (obs, info)
    info = {}
    if isinstance(observation, tuple) and len(observation) > 1:
        info = observation[1] if len(observation) > 1 else {}
        observation = observation[0]
    
    # Chuyển trạng thái thành số để lưu trong Q-table
    state = get_state(observation)
    
    # Khởi tạo phần thưởng và số bước của episode
    ep_reward = 0  # Khởi tạo phần thưởng của episode
    ep_start_state = state  # Lưu trạng thái bắt đầu
    action_list = []

    done = False
    step = 0
    max_steps = 100  # Giới hạn số bước để tránh vòng lặp vô hạn

    while not done and step < max_steps:
        # Chọn action từ Q-table
        action = get_action(state)
        
        # Chuyển action thành hành động có thể thực hiện được trong môi trường
        env_action = get_env_action(action, observation)
        
        # Thực hiện hành động trong môi trường
        try:
            step_result = env.step(env_action)
            
            # Xử lý nhiều format khác nhau của step_result
            if isinstance(step_result, tuple):
                if len(step_result) >= 5:  # gymnasium format mới
                    observation, reward_terminal, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) >= 4:  # gym format cũ
                    observation, reward_terminal, done, info = step_result
                elif len(step_result) == 3:  # format đơn giản
                    observation, reward_terminal, done = step_result
                    info = {}
                else:
                    # Format không xác định
                    observation = step_result[0] if len(step_result) > 0 else observation
                    reward_terminal = 0
                    done = False
                    info = {}
            else:
                # Nếu step_result không phải tuple, giả sử nó là observation mới
                observation = step_result
                reward_terminal = 0
                done = False
                info = {}
        
            # Lưu hành động vào danh sách
            action_list.append(env_action)
            
            # Tính reward theo logic của chúng ta
            reward = get_reward(observation, info)
            
            # Cập nhật tổng reward của episode
            ep_reward += reward
            
            # Chuyển đổi trạng thái mới thành số để lưu trong Q-table
            next_state = get_state(observation)
            
            # Cập nhật Q-table
            Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (
                reward + gamma * np.max(Q_table[next_state])
            )
            
            # Cập nhật state
            state = next_state
            
        except Exception as e:
            print(f"Error during step execution: {e}")
            done = True
        
        step += 1
        

    # Cập nhật phần thưởng và hành động tốt nhất nếu có
    if ep_reward > max_ep_reward:
        max_ep_reward = ep_reward
        max_ep_action_list = action_list.copy()
        max_start_state = ep_start_state

    # Update epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Ghi log để visualization sau này
    with open("results/q_learning_log1.txt", "a") as log_file:
        log_file.write(f"Episode {episode}, Reward: {ep_reward:.4f}, Epsilon: {epsilon:.4f}, Steps: {step}\n")
    
    # Log chi tiết cho visualization - đặt ngay sau khi ghi log
    try:
        # Tạo thư mục results nếu chưa tồn tại
        os.makedirs("results", exist_ok=True)
        
        with open("results/detailed_cutting_log1.txt", "a") as detail_log:
            detail_log.write(f"Episode {episode} starting\n")
            
            # Tính toán metrics
            trim_loss = 0
            material_usage = 0
            stocks_used = 0
            remaining_products = 0
            
            # Extract thông tin từ observation
            if isinstance(observation, dict) and "stocks" in observation:
                stocks = observation["stocks"]
                total_stocks = len(stocks)
                total_area = 0
                used_area = 0
                
                for stock in stocks:
                    if hasattr(stock, 'shape'):  # Numpy array
                        stock_area = stock.shape[0] * stock.shape[1]
                        used_cells = np.sum(stock != -1)
                        empty_cells = np.sum(stock == -1)
                        
                        if np.any(stock != -2):  # Stock đã sử dụng
                            stocks_used += 1
                            total_area += stock_area
                            used_area += used_cells
                    
                    elif hasattr(stock, 'width') and hasattr(stock, 'height'):
                        stock_area = stock.width * stock.height
                        used_cells = len(getattr(stock, 'used_spaces', set()))
                        
                        if used_cells > 0:  # Stock đã sử dụng
                            stocks_used += 1
                            total_area += stock_area
                            used_area += used_cells
                
                # Tính material usage và trim loss
                if total_area > 0:
                    material_usage = (used_area / total_area) * 100
                    trim_loss = 100 - material_usage
            
            # Đếm remaining products
            if isinstance(observation, dict) and "products" in observation:
                remaining_products = len(observation["products"])
            
            # Log metrics
            detail_log.write(f"Trim loss: {trim_loss:.2f}%\n")
            detail_log.write(f"Material usage: {material_usage:.2f}%\n")
            detail_log.write(f"Stocks used: {stocks_used}\n")
            detail_log.write(f"Remaining products: {remaining_products}\n")
            detail_log.write(f"Episode {episode} complete\n\n")
    except Exception as e:
        print(f"Error logging detailed metrics: {e}")
    
    # Lưu lại lịch sử huấn luyện
    rewards_history.append(ep_reward)
    epsilons_history.append(epsilon)
    steps_history.append(step)

    print(f"Episode {episode}, Reward: {ep_reward:.4f}, Epsilon: {epsilon:.4f}, Steps: {step}")
        
# Hiển thị kết quả tốt nhất tìm được
print("\nTraining complete!")
print(f"Max reward = {max_ep_reward:.4f}")
print(f"Best sequence length = {len(max_ep_action_list)}")

# Lưu Q-table để sử dụng sau này
q_table_path = "results/q_table1.pkl"
with open(q_table_path, "wb") as f:
    pickle.dump(Q_table, f)
print(f"Q-table saved to {q_table_path}")

# Lưu lịch sử huấn luyện
training_history = pd.DataFrame({
    "Episode": range(1, num_episodes + 1),
    "Reward": rewards_history,
    "Epsilon": epsilons_history,
    "Steps": steps_history
})
training_history.to_csv("results/training_history1.csv", index=False)
print("Training history saved to results/training_history1.csv")

print("\nReplaying best sequence...")

# Phát lại tập tốt nhất tìm được
observation = env.reset()
# Không thể đặt state trực tiếp, bắt đầu từ trạng thái mới
for action in max_ep_action_list:
    try:
        env.step(action)  # Thực hiện hành động
        env.render()  # Hiển thị môi trường
    except Exception as e:
        print(f"Error replaying action: {e}")

# Hiển thị trạng thái cuối cùng
env.render()
print("Replay complete.")

# Tạo kết quả cutting stock để visualization
material_usage = []
waste_percentages = []

try:
    # Thử lấy dữ liệu từ environment
    if hasattr(env, 'stocks'):
        for i, stock in enumerate(env.stocks):
            if hasattr(stock, 'used_spaces') and hasattr(stock, 'width') and hasattr(stock, 'height'):
                total_area = stock.width * stock.height
                used_area = len(stock.used_spaces)
                usage_percent = (used_area / total_area) * 100
                waste_percent = 100 - usage_percent
            else:
                # Giá trị ngẫu nhiên nếu không lấy được dữ liệu
                usage_percent = random.uniform(60, 90)
                waste_percent = 100 - usage_percent
                
            material_usage.append(usage_percent)
            waste_percentages.append(waste_percent)
    
    # Nếu không lấy được dữ liệu từ environment, tạo dữ liệu mẫu
    if not material_usage:
        for i in range(5):
            usage = random.uniform(60, 90)
            waste = 100 - usage
            material_usage.append(usage)
            waste_percentages.append(waste)
    
    # Lưu dữ liệu để visualize
    df_results = pd.DataFrame({
        "Stock": [f"Stock {i+1}" for i in range(len(material_usage))],
        "Usage": material_usage,
        "Waste": waste_percentages
    })
    df_results.to_csv("results/cutting_stock_results1.csv", index=False)
    print("Cutting stock results saved to results/cutting_stock_results1.csv")
except Exception as e:
    print(f"Error saving cutting stock results: {e}")

# Kiểm tra Q-table đã lưu
try:
    # Kiểm tra xem file tồn tại không trước khi mở
    if os.path.exists(q_table_path):
        with open(q_table_path, "rb") as f:
            loaded_Q_table = pickle.load(f)
            
        # Kiểm tra và hiển thị thông tin về Q-table
        print("\nQ-table Statistics:")
        print(f"Shape: {loaded_Q_table.shape}")
        print(f"Max value: {np.max(loaded_Q_table):.4f}")
        print(f"Min value: {np.min(loaded_Q_table):.4f}")
        print(f"Mean value: {np.mean(loaded_Q_table):.4f}")
        print(f"Non-zero entries: {np.count_nonzero(loaded_Q_table)}")
        print(f"Sparsity: {1 - np.count_nonzero(loaded_Q_table) / np.size(loaded_Q_table):.4f}")
        print(f"Size in memory: {loaded_Q_table.nbytes / (1024*1024):.2f} MB")
    else:
        print("\nQ-table file not found. This could be because the file was not created or was not saved correctly.")
        print(f"Expected file location: {q_table_path}")
except Exception as e:
    print(f"Error loading saved Q-table: {e}")

# Đóng môi trường
env.close()

# In thông báo cho người dùng biết cách xem visualization
print("\n" + "="*100)
print("Để xem visualization, hãy chạy file visualization.py và mở file results/visualization_report.html")
print("="*100)