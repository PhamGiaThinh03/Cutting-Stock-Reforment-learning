# env/actions.py
import numpy as np
from gymnasium import spaces

def setup_spaces(env):
    # Không gian quan sát
    upper = np.full(shape=(env.max_w, env.max_h), fill_value=env.max_product_type + 2, dtype=int)
    lower = np.full(shape=(env.max_w, env.max_h), fill_value=-2, dtype=int)
    observation_space = spaces.Dict(
        {
            "stocks": spaces.Tuple(
                [spaces.MultiDiscrete(upper, start=lower)] * env.num_stocks, seed=env.seed
            ),
            "products": spaces.Sequence(
                spaces.Dict(
                    {
                        "size": spaces.MultiDiscrete(
                            np.array([env.max_w, env.max_h]), start=np.array([1, 1])
                        ),
                        "quantity": spaces.Discrete(env.max_product_per_type + 1, start=0),
                    }
                ),
                seed=env.seed,
            ),
        }
    )

    # Không gian hành động
    action_space = spaces.Dict(
        {
            "stock_idx": spaces.Discrete(env.num_stocks),
            "size": spaces.Box(
                low=np.array([1, 1]),
                high=np.array([env.max_w, env.max_h]),
                shape=(2,),
                dtype=int,
            ),
            "position": spaces.Box(
                low=np.array([0, 0]),
                high=np.array([env.max_w - 1, env.max_h - 1]),
                shape=(2,),
                dtype=int,
            ),
        }
    )

    return observation_space, action_space

def get_valid_actions(env):
    """
    Get all valid actions for the current environment state.
    
    Args:
        env: The cutting stock environment
        
    Returns:
        list: A list of valid action dictionaries
    """
    valid_actions = []
    
    # Try to get state via different methods
    try:
        if hasattr(env, 'get_state'):
            state = env.get_state()
        else:
            # Directly work with environment attributes
            state = {
                'current_stock_id': getattr(env, 'current_stock_id', 0),
                'products': getattr(env, 'products', []),
                'stocks': getattr(env, 'stocks', {})
            }
        
        # Debug state
        print(f"State type: {type(state)}")
        if isinstance(state, dict):
            print(f"State keys: {state.keys()}")
    except Exception as e:
        print(f"Error getting state: {str(e)}")
        return []
    
    # Extract state information safely
    if not isinstance(state, dict):
        print(f"State is not a dictionary: {type(state)}")
        return []
    
    current_stock_id = state.get('current_stock_id', 0)
    products = state.get('products', [])
    stocks = state.get('stocks', {})
    
    # If no remaining products or no stocks, no valid actions
    if not products or current_stock_id not in stocks:
        return valid_actions
    
    current_stock = stocks[current_stock_id]
    
    # Check if current_stock has required attributes
    if not hasattr(current_stock, 'width') or not hasattr(current_stock, 'height'):
        print(f"Current stock missing width/height attributes")
        return []
    
    # Check each product
    for product_idx, product in enumerate(products):
        if not hasattr(product, 'width') or not hasattr(product, 'height'):
            continue
            
        product_width = product.width
        product_height = product.height
        
        # Skip if product is larger than stock
        if product_width > current_stock.width or product_height > current_stock.height:
            continue
        
        # Check all possible positions
        for i in range(current_stock.width - product_width + 1):
            for j in range(current_stock.height - product_height + 1):
                # Check if the product can be placed at position (i, j)
                can_place = True
                
                # Check for used_spaces attribute
                if not hasattr(current_stock, 'used_spaces'):
                    # Assume empty stock if no used_spaces attribute
                    used_spaces = set()
                else:
                    used_spaces = current_stock.used_spaces
                
                # Check if any part of the position is already occupied
                for x in range(i, i + product_width):
                    for y in range(j, j + product_height):
                        if (x, y) in used_spaces:
                            can_place = False
                            break
                    if not can_place:
                        break
                
                if can_place:
                    # Create valid action
                    action = {
                        "stock_idx": current_stock_id,
                        "size": np.array([product_width, product_height], dtype=int),
                        "position": np.array([i, j], dtype=int),
                        "product_id": getattr(product, 'id', product_idx)  # Use index if no id attribute
                    }
                    valid_actions.append(action)
    
    # Print count of valid actions
    print(f"Found {len(valid_actions)} valid actions")
    return valid_actions