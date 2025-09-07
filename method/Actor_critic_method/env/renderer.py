# env/renderer.py
import pygame
import numpy as np
import matplotlib as mpl
from matplotlib import colormaps

def get_window_size(env):
    width = int(np.ceil(np.sqrt(env.num_stocks)))
    height = int(np.ceil(env.num_stocks / width))
    return width * env.max_w, height * env.max_h

def render_frame(env):
    window_size = get_window_size(env)
    if env.window is None and env.render_mode == "human":
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption("Cutting Stock Environment")
        env.window = pygame.display.set_mode(window_size)
    if env.clock is None and env.render_mode == "human":
        env.clock = pygame.time.Clock()
    
    canvas = pygame.Surface(window_size)
    canvas.fill((0, 0, 0))
    pix_square_size = 1
    
    # Tạo danh sách màu
    cmap = colormaps.get_cmap("hsv")
    norms = mpl.colors.Normalize(vmin=0, vmax=env.max_product_type - 1)
    list_colors = [cmap(norms(i))[:3] for i in range(env.max_product_type)]
    list_colors.extend([(1, 1, 1)] * 10)  # Thêm màu trắng để tránh lỗi
    
    for i, stock in enumerate(env._stocks):
        stock_width = int(np.sum(np.any(stock != -2, axis=1)))
        stock_height = int(np.sum(np.any(stock != -2, axis=0)))
        pygame.draw.rect(
            canvas,
            (128, 128, 128),
            pygame.Rect(
                (i % (window_size[0] // env.max_w) * env.max_w) * pix_square_size,
                (i // (window_size[0] // env.max_w) * env.max_h) * pix_square_size,
                stock_width * pix_square_size,
                stock_height * pix_square_size,
            ),
        )
        
        for x in range(stock.shape[0]):
            for y in range(stock.shape[1]):
                if stock[x, y] > -1:
                    if 0 <= stock[x, y] < len(list_colors):
                        color = list_colors[stock[x, y]]
                    else:
                        color = (1, 1, 1)
                    
                    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                    
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(
                            (i % (window_size[0] // env.max_w) * env.max_w + x) * pix_square_size,
                            (i // (window_size[0] // env.max_w) * env.max_h + y) * pix_square_size,
                            pix_square_size,
                            pix_square_size,
                        ),
                    )
    
    # Vẽ lưới
    for i in range(window_size[0] // env.max_w):
        pygame.draw.line(
            canvas, (255, 255, 255),
            (i * env.max_w * pix_square_size, 0),
            (i * env.max_w * pix_square_size, window_size[1]),
        )
    for i in range(window_size[1] // env.max_h):
        pygame.draw.line(
            canvas, (255, 255, 255),
            (0, i * env.max_h * pix_square_size),
            (window_size[0], i * env.max_h * pix_square_size),
        )
    
    if env.render_mode == "human":
        env.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        env.clock.tick(env.metadata["render_fps"])
    else:
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))