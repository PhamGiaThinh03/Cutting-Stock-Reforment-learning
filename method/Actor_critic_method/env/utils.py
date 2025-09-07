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
    for sid, stock in enumerate(env._stocks):
        if env.cutted_stocks[sid] == 0:
            continue
        tl = (stock == -1).sum() / (stock != -2).sum()
        trim_loss.append(tl)
    trim_loss = np.mean(trim_loss).item() if trim_loss else 1
    return {"filled_ratio": filled_ratio, "trim_loss": trim_loss}