# env/utils.py
import numpy as np

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

def get_obs(env):
    return {"stocks": env._stocks, "products": env._products}

def get_info(env):
    filled_ratio = np.mean(env.cutted_stocks).item()
    trim_loss = []
    used_stocks = 0
    for sid, stock in enumerate(env._stocks):
        if env.cutted_stocks[sid] == 0:
            continue
        tl = (stock == -1).sum() / (stock != -2).sum()
        trim_loss.append(tl)
        used_stocks += 1
    trim_loss = np.mean(trim_loss).item() if trim_loss else 1
    # Tính số lượng sản phẩm còn lại
    remaining_products = sum(p["quantity"] for p in env._products)
    return {
        "filled_ratio": filled_ratio,
        "trim_loss": trim_loss,
        "used_stocks": used_stocks,
        "remaining_products": remaining_products
    }