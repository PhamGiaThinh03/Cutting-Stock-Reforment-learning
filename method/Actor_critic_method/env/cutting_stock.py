import gymnasium as gym
import numpy as np
from .constants import METADATA
from .utils import set_seed, get_obs, get_info
from .actions import setup_spaces
from .renderer import render_frame, get_window_size

class CuttingStockEnv(gym.Env):
    metadata = METADATA

    def __init__(
        self,
        render_mode=None,
        min_w=50,
        min_h=50,
        max_w=100,
        max_h=100,
        num_stocks=100,
        max_product_type=25,
        max_product_per_type=20,
        seed=42,
        stock_list=None,
        product_list=None,
        data_path=None,
    ):
        self.seed = seed
        set_seed(seed)
        
        self.min_w = min_w
        self.min_h = min_h
        self.max_w = max_w
        self.max_h = max_h

        if stock_list is not None:
            self.stock_list = stock_list
            self.num_stocks = len(stock_list)
        else:
            self.stock_list = None
            self.num_stocks = num_stocks

        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        
        self.product_list = product_list

        self.observation_space, self.action_space = setup_spaces(self)

        self._stocks = []
        self._products = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        set_seed(seed)
        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        
        self._stocks = []
        if self.stock_list is not None:
            for (w, h) in self.stock_list:
                stock = np.full((self.max_w, self.max_h), fill_value=-2, dtype=int)
                stock[:w, :h] = -1
                self._stocks.append(stock)
        else:
            for _ in range(self.num_stocks):
                width = np.random.randint(low=self.min_w, high=self.max_w + 1)
                height = np.random.randint(low=self.min_h, high=self.max_h + 1)
                stock = np.full((self.max_w, self.max_h), fill_value=-2, dtype=int)
                stock[:width, :height] = -1
                self._stocks.append(stock)
        self._stocks = tuple(self._stocks)
        
        self._products = []
        if self.product_list is not None:
            for (w, h) in self.product_list:
                product = {"size": np.array([w, h]), "quantity": 1}
                self._products.append(product)
        else:
            num_type_products = np.random.randint(low=1, high=self.max_product_type)
            for _ in range(num_type_products):
                width = np.random.randint(low=1, high=self.min_w + 1)
                height = np.random.randint(low=1, high=self.min_h + 1)
                quantity = np.random.randint(low=1, high=self.max_product_per_type + 1)
                product = {"size": np.array([width, height]), "quantity": quantity}
                self._products.append(product)
        self._products = tuple(self._products)

        observation = get_obs(self)
        info = get_info(self)

        if self.render_mode == "human":
            render_frame(self)
        return observation, info
    
    def step(self, action):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        width, height = size
        x, y = position

        product_idx = None
        for i, product in enumerate(self._products):
            if np.array_equal(product["size"], size) or np.array_equal(
                product["size"], size[::-1]
            ):
                if product["quantity"] == 0:
                    continue
                product_idx = i
                break

        if product_idx is not None:
            if 0 <= stock_idx < self.num_stocks:
                stock = self._stocks[stock_idx]
                stock_width = np.sum(np.any(stock != -2, axis=1))
                stock_height = np.sum(np.any(stock != -2, axis=0))
                if (
                    x >= 0
                    and y >= 0
                    and x + width <= stock_width
                    and y + height <= stock_height
                ):
                    if np.all(stock[x : x + width, y : y + height] == -1):
                        self.cutted_stocks[stock_idx] = 1
                        stock[x : x + width, y : y + height] = product_idx
                        self._products[product_idx]["quantity"] -= 1

        terminated = all([product["quantity"] == 0 for product in self._products])
        reward = 1 if terminated else 0

        observation = get_obs(self)
        info = get_info(self)

        if self.render_mode == "human":
            render_frame(self)

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return render_frame(self)

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            self.window = None
            self.clock = None

    def _get_window_size(self):
        return get_window_size(self)
    
# import random
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces
# import pygame
# import matplotlib as mpl
# from matplotlib import colormaps

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)

# class CuttingStockEnv(gym.Env):
#     """
#     Môi trường Cutting Stock với khả năng khởi tạo stocks và products theo danh sách do người dùng truyền vào.
    
#     Các tham số mới:
#       - stock_list: Danh sách các stock, mỗi stock là một tuple (width, height).
#       - product_list: Danh sách các product, mỗi product là một tuple (width, height). Mặc định mỗi sản phẩm có số lượng cần cắt là 1.
      
#     Nếu không truyền vào các list này, môi trường sẽ tạo dữ liệu ngẫu nhiên
#     """
    
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}

#     def __init__(
#         self,
#         render_mode=None,
#         min_w=50,
#         min_h=50,
#         max_w=100,
#         max_h=100,
#         num_stocks=100,
#         max_product_type=25,
#         max_product_per_type=20,
#         seed=42,
#         stock_list=None,      # Danh sách stock dạng [(w1, h1), (w2, h2), ...]
#         product_list=None,    # Danh sách product dạng [(w1, h1), (w2, h2), ...]
#     ):
#         self.seed = seed
#         set_seed(seed)
        
#         self.min_w = min_w
#         self.min_h = min_h
#         self.max_w = max_w
#         self.max_h = max_h

#         # Nếu truyền stock_list, cập nhật số stock theo đó
#         if stock_list is not None:
#             self.stock_list = stock_list
#             self.num_stocks = len(stock_list)
#         else:
#             self.stock_list = None
#             self.num_stocks = num_stocks

#         self.max_product_type = max_product_type
#         self.max_product_per_type = max_product_per_type
#         self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        
#         # Nếu truyền product_list, lưu lại
#         self.product_list = product_list

#         # Thiết lập không gian quan sát (không thay đổi nhiều)
#         upper = np.full(shape=(max_w, max_h), fill_value=max_product_type + 2, dtype=int)
#         lower = np.full(shape=(max_w, max_h), fill_value=-2, dtype=int)
#         self.observation_space = spaces.Dict(
#             {
#                 "stocks": spaces.Tuple(
#                     [spaces.MultiDiscrete(upper, start=lower)] * self.num_stocks, seed=seed
#                 ),
#                 "products": spaces.Sequence(
#                     spaces.Dict(
#                         {
#                             "size": spaces.MultiDiscrete(
#                                 np.array([max_w, max_h]), start=np.array([1, 1])
#                             ),
#                             "quantity": spaces.Discrete(max_product_per_type + 1, start=0),
#                         }
#                     ),
#                     seed=seed,
#                 ),
#             }
#         )

#         # Không gian hành động như cũ
#         self.action_space = spaces.Dict(
#             {
#                 "stock_idx": spaces.Discrete(self.num_stocks),
#                 "size": spaces.Box(
#                     low=np.array([1, 1]),
#                     high=np.array([max_w, max_h]),
#                     shape=(2,),
#                     dtype=int,
#                 ),
#                 "position": spaces.Box(
#                     low=np.array([0, 0]),
#                     high=np.array([max_w - 1, max_h - 1]),
#                     shape=(2,),
#                     dtype=int,
#                 ),
#             }
#         )

#         # Khởi tạo danh sách stocks và products rỗng
#         self._stocks = []
#         self._products = []

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

#         self.window = None
#         self.clock = None

#     def _get_obs(self):
#         return {"stocks": self._stocks, "products": self._products}

#     def _get_info(self):
#         filled_ratio = np.mean(self.cutted_stocks).item()
#         trim_loss = []
#         for sid, stock in enumerate(self._stocks):
#             if self.cutted_stocks[sid] == 0:
#                 continue
#             tl = (stock == -1).sum() / (stock != -2).sum()
#             trim_loss.append(tl)
#         trim_loss = np.mean(trim_loss).item() if trim_loss else 1
#         return {"filled_ratio": filled_ratio, "trim_loss": trim_loss}

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         set_seed(seed)
#         self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)
        
#         # Khởi tạo stocks:
#         self._stocks = []
#         if self.stock_list is not None:
#             # Sử dụng danh sách stock do người dùng truyền vào.
#             for (w, h) in self.stock_list:
#                 # Tạo một mảng kích thước (max_w, max_h) với giá trị -2 bên ngoài vùng stock.
#                 stock = np.full((self.max_w, self.max_h), fill_value=-2, dtype=int)
#                 # Vùng stock thực có kích thước (w, h) được đánh dấu là trống (-1)
#                 stock[:w, :h] = -1
#                 self._stocks.append(stock)
#         else:
#             # Nếu không truyền stock_list, tạo ngẫu nhiên như cũ.
#             for _ in range(self.num_stocks):
#                 width = np.random.randint(low=self.min_w, high=self.max_w + 1)
#                 height = np.random.randint(low=self.min_h, high=self.max_h + 1)
#                 stock = np.full((self.max_w, self.max_h), fill_value=-2, dtype=int)
#                 stock[:width, :height] = -1
#                 self._stocks.append(stock)
#         self._stocks = tuple(self._stocks)
        
#         # Khởi tạo products:
#         self._products = []
#         if self.product_list is not None:
#             # Với mỗi product, giả định số lượng cần cắt là 1 (có thể chỉnh sửa nếu cần)
#             for (w, h) in self.product_list:
#                 product = {"size": np.array([w, h]), "quantity": 1}
#                 self._products.append(product)
#         else:
#             # Sinh ngẫu nhiên danh sách product như cũ.
#             num_type_products = np.random.randint(low=1, high=self.max_product_type)
#             for _ in range(num_type_products):
#                 width = np.random.randint(low=1, high=self.min_w + 1)
#                 height = np.random.randint(low=1, high=self.min_h + 1)
#                 quantity = np.random.randint(low=1, high=self.max_product_per_type + 1)
#                 product = {"size": np.array([width, height]), "quantity": quantity}
#                 self._products.append(product)
#         self._products = tuple(self._products)

#         observation = self._get_obs()
#         info = self._get_info()

#         if self.render_mode == "human":
#             self._render_frame()
#         return observation, info
    
#     def step(self, action):
#         stock_idx = action["stock_idx"]
#         size = action["size"]
#         position = action["position"]

#         width, height = size
#         x, y = position

#         # Check if the product is in the product list
#         product_idx = None
#         for i, product in enumerate(self._products):
#             if np.array_equal(product["size"], size) or np.array_equal(
#                 product["size"], size[::-1]
#             ):
#                 if product["quantity"] == 0:
#                     continue

#                 product_idx = i  # Product index starts from 0
#                 break

#         if product_idx is not None:
#             if 0 <= stock_idx < self.num_stocks:
#                 stock = self._stocks[stock_idx]
#                 # Check if the product fits in the stock
#                 stock_width = np.sum(np.any(stock != -2, axis=1))
#                 stock_height = np.sum(np.any(stock != -2, axis=0))
#                 if (
#                     x >= 0
#                     and y >= 0
#                     and x + width <= stock_width
#                     and y + height <= stock_height
#                 ):
#                     # Check if the position is empty
#                     if np.all(stock[x : x + width, y : y + height] == -1):
#                         self.cutted_stocks[stock_idx] = 1
#                         stock[x : x + width, y : y + height] = product_idx
#                         self._products[product_idx]["quantity"] -= 1

#         # An episode is done iff the all product quantities are 0
#         terminated = all([product["quantity"] == 0 for product in self._products])
#         reward = 1 if terminated else 0  # Binary sparse rewards

#         observation = self._get_obs()
#         info = self._get_info()

#         if self.render_mode == "human":
#             self._render_frame()

#         return observation, reward, terminated, False, info

#     def render(self):
#         if self.render_mode == "rgb_array":
#             return self._render_frame()

#     def _get_window_size(self):
#         width = int(np.ceil(np.sqrt(self.num_stocks)))
#         height = int(np.ceil(self.num_stocks / width))
#         return width * self.max_w, height * self.max_h

#     def _render_frame(self):
#         window_size = self._get_window_size()
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             pygame.display.set_caption("Cutting Stock Environment")
#             self.window = pygame.display.set_mode(window_size)
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()
        
#         canvas = pygame.Surface(window_size)
#         canvas.fill((0, 0, 0))
#         pix_square_size = 1
        
#         # Tạo danh sách màu, thêm phần tử dư phòng tránh lỗi IndexError
#         cmap = colormaps.get_cmap("hsv")
#         norms = mpl.colors.Normalize(vmin=0, vmax=self.max_product_type - 1)
#         list_colors = [cmap(norms(i))[:3] for i in range(self.max_product_type)]
#         list_colors.extend([(1, 1, 1)] * 10)  # Thêm màu trắng để tránh lỗi index
        
#         for i, stock in enumerate(self._stocks):
#             stock_width = int(np.sum(np.any(stock != -2, axis=1)))
#             stock_height = int(np.sum(np.any(stock != -2, axis=0)))
#             pygame.draw.rect(
#                 canvas,
#                 (128, 128, 128),
#                 pygame.Rect(
#                     (i % (window_size[0] // self.max_w) * self.max_w) * pix_square_size,
#                     (i // (window_size[0] // self.max_w) * self.max_h) * pix_square_size,
#                     stock_width * pix_square_size,
#                     stock_height * pix_square_size,
#                 ),
#             )
            
#             for x in range(stock.shape[0]):
#                 for y in range(stock.shape[1]):
#                     if stock[x, y] > -1:
#                         # **Sửa lỗi index bằng cách kiểm tra giới hạn**
#                         if 0 <= stock[x, y] < len(list_colors):
#                             color = list_colors[stock[x, y]]
#                         else:
#                             color = (1, 1, 1)  # Màu trắng nếu lỗi
                        
#                         color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                        
#                         pygame.draw.rect(
#                             canvas,
#                             color,
#                             pygame.Rect(
#                                 (i % (window_size[0] // self.max_w) * self.max_w + x) * pix_square_size,
#                                 (i // (window_size[0] // self.max_w) * self.max_h + y) * pix_square_size,
#                                 pix_square_size,
#                                 pix_square_size,
#                             ),
#                         )
        
#         # Vẽ lưới
#         for i in range(window_size[0] // self.max_w):
#             pygame.draw.line(
#                 canvas, (255, 255, 255),
#                 (i * self.max_w * pix_square_size, 0),
#                 (i * self.max_w * pix_square_size, window_size[1]),
#             )
#         for i in range(window_size[1] // self.max_h):
#             pygame.draw.line(
#                 canvas, (255, 255, 255),
#                 (0, i * self.max_h * pix_square_size),
#                 (window_size[0], i * self.max_h * pix_square_size),
#             )
        
#         if self.render_mode == "human":
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()
#             self.clock.tick(self.metadata["render_fps"])
#         else:
#             return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))


#     def close(self):
#         if self.window is not None:
#             pygame.display.quit()
#             pygame.font.quit()
# env/environment.py
