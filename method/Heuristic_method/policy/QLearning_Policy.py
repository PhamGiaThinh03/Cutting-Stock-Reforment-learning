import numpy as np
import random
import pickle
import os
import sys
from collections import defaultdict

# Add parent directory to sys.path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import get_valid_actions from actions module
from env.actions import get_valid_actions

class QLearningPolicy:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, 
             exploration_decay=0.995, min_exploration_rate=0.01, save_path=None):
        self.env = env
        self.learning_rate = learning_rate        # Alpha
        self.discount_factor = discount_factor    # Gamma
        self.exploration_rate = exploration_rate  # Epsilon
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Use defaultdict for q_table
        print("Initializing new Q-table")
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        self.save_path = save_path or "q_table.pkl"
        
        # For tracking performance
        self.total_rewards = []
        self.episode_lengths = []
        
        # Add get_state method to environment if needed
        if not hasattr(env, 'get_state'):
            self._add_get_state_to_env()
        
    def state_to_key(self, state):
        """Convert state to a hashable representation for Q-table"""
        try:
            # Extract relevant state information in a safe way
            if isinstance(state, dict):
                stock_id = state.get('current_stock_id', 0)
                
                # Handle products based on the format
                products = state.get('products', [])
                if hasattr(products, '__iter__'):
                    # Try to get product dimensions and IDs
                    try:
                        remaining_products = frozenset([(p.width, p.height, p.id) 
                                                       for p in products])
                    except AttributeError:
                        # Fallback if products have a different structure
                        remaining_products = frozenset([(i,) for i in range(len(products))])
                else:
                    remaining_products = frozenset()
                
                # Get used spaces if available
                stocks = state.get('stocks', {})
                if stock_id in stocks and hasattr(stocks[stock_id], 'used_spaces'):
                    used_spaces = frozenset(tuple(space) for space in stocks[stock_id].used_spaces)
                else:
                    used_spaces = frozenset()
                
                # Create a hashable state representation
                return (stock_id, remaining_products, used_spaces)
            else:
                # If state is not a dict, use a simple representation
                return hash(str(state))
        except Exception as e:
            print(f"Error in state_to_key: {e}")
            # Return a default key in case of error
            return (0, frozenset(), frozenset())
    
    def choose_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        # Get state if not provided
        if state is None:
            try:
                state = self.env.get_state()
            except AttributeError:
                state = {}  # Use empty state if get_state not available
        
        # Get valid actions
        try:
            valid_actions = get_valid_actions(self.env)
        except Exception as e:
            print(f"Error getting valid actions: {e}")
            return None  # Return None if no valid actions
            
        if not valid_actions:
            return None  # No valid actions available
            
        state_key = self.state_to_key(state)
        
        # Exploration: random action
        if training and random.random() < self.exploration_rate:
            return random.choice(valid_actions)
        
        # Exploitation: best known action
        # If all q-values are 0, choose randomly
        if not any(self.q_table[state_key][str(a)] for a in valid_actions):
            return random.choice(valid_actions)
        
        # Choose action with highest Q-value
        # Convert action dict to string for dictionary key
        return max(valid_actions, key=lambda a: self.q_table[state_key][str(a)])
    
    def update_q_value(self, state, action, next_state, reward):
        """Update Q-value for a state-action pair"""
        # Skip update if any parameter is None
        if state is None or action is None or next_state is None:
            return
            
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        action_key = str(action)  # Convert action dict to string for dictionary key
        
        # Get best Q-value for next state
        try:
            next_valid_actions = get_valid_actions(self.env)
            best_next_q = max([self.q_table[next_state_key][str(a)] for a in next_valid_actions]) if next_valid_actions else 0
        except Exception:
            best_next_q = 0  # Default to 0 if there's an error
        
        # Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        self.q_table[state_key][action_key] += self.learning_rate * (
            reward + 
            self.discount_factor * best_next_q - 
            self.q_table[state_key][action_key]
        )
    
    def select_action(self, env):
        """Interface method for compatibility with other policies"""
        # Get state from environment
        try:
            state = env.get_state()
        except AttributeError:
            state = None
            
        return self.choose_action(state, training=False)
    
    def save_policy(self):
        """Save the Q-table to disk"""
        # Convert defaultdict to dict for serialization
        q_dict = {}
        for k, actions in self.q_table.items():
            q_dict[str(k)] = {str(a): v for a, v in actions.items()}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        with open(self.save_path, 'wb') as f:
            pickle.dump(q_dict, f)
        print(f"Policy saved to {self.save_path}")
    
    # Thêm vào phương thức load_policy()

def save_policy(self):
    """Save the Q-table to disk with improved error handling."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Convert defaultdict to regular dict for serialization
        q_dict = {}
        for state, actions in self.q_table.items():
            q_dict[str(state)] = dict(actions)
        
        # Check if dictionary is empty
        if not q_dict:
            print("Warning: Attempted to save empty Q-table!")
            # Add dummy data to prevent empty file
            q_dict["dummy_state"] = {"dummy_action": 0.0}
        
        with open(self.save_path, 'wb') as f:
            pickle.dump(q_dict, f)
        
        print(f"Q-table saved to {self.save_path} with {len(q_dict)} states")
        return True
    except Exception as e:
        print(f"Error saving Q-table: {e}")
        return False

# For testing
if __name__ == "__main__":
    print("QLearningPolicy module loaded successfully")