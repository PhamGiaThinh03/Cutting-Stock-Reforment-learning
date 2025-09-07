################## Ý tưởng ##################

# First-Fit Placement:
#         Sử dụng thuật toán First-Fit để đặt nhanh các sản phẩm vào stock có sẵn.
#         Mục tiêu: Tốc độ cao, giảm thời gian tính toán.
        
# Best-Fit Refinement:
#         Sau khi First-Fit đặt xong, sử dụng Best-Fit để điều chỉnh vị trí sản phẩm.
#         Mục tiêu: Tối ưu hóa diện tích trống, giảm lãng phí vật liệu.
        
# Stock Merging:
#     Nếu một stock đã cắt nhỏ hơn một stock chưa cắt, di chuyển phần đã cắt vào stock lớn hơn để giảm số lượng stock cần dùng.

import numpy as np

def first_fit_placement(observation):
    """
    Phase 1: First-Fit Placement - Đặt sản phẩm vào vị trí có thể chứa nó nhanh nhất.
    """
    list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
    list_stocks = sorted(observation["stocks"], key=lambda x: np.sum(x != -2), reverse=True)

    for product_idx, prod in enumerate(list_prods):  # Dùng product_idx để phân biệt màu
        if prod["quantity"] <= 0:
            continue  # Bỏ qua sản phẩm đã hết

        prod_w, prod_h = prod["size"]

        for idx, stock in enumerate(list_stocks):
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))

            if stock_w < prod_w or stock_h < prod_h:
                continue  

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                        stock[x:x + prod_w, y:y + prod_h] = product_idx + 2  # Sử dụng product_idx + 2
                        prod["quantity"] -= 1  # Giảm số lượng sản phẩm
                        return {"stock_idx": idx, "size": (prod_w, prod_h), "position": (x, y)}

    return None

 


def best_fit_refinement(observation):
    """
    Phase 2: Best-Fit Refinement - Tìm vị trí có `Sij` nhỏ nhất.
    """
    list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
    list_stocks = sorted(observation["stocks"], key=lambda x: np.sum(x != -2), reverse=True)

    best_action = None
    min_sij = float("inf")

    for product_idx, prod in enumerate(list_prods):  # Sử dụng index sản phẩm để phân biệt màu
        if prod["quantity"] <= 0:
            continue  

        prod_w, prod_h = prod["size"]

        for stock_idx, stock in enumerate(list_stocks):
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))

            if stock_w < prod_w or stock_h < prod_h:
                continue  

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                        sij = (x + prod_w) * (y + prod_h)  # Diện tích nhỏ nhất chứa sản phẩm

                        if sij < min_sij:
                            best_action = {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x, y)}
                            min_sij = sij
                            stock[x:x + prod_w, y:y + prod_h] = product_idx + 2  # Gán ID riêng
                            prod["quantity"] -= 1

    return best_action




def stock_merging(observation):
    """
    Phase 3: Stock Merging - Hợp nhất stock nhỏ vào stock lớn hơn.
    """
    list_stocks = sorted(observation["stocks"], key=lambda x: np.sum(x != -2)) 

    for i, stock_i in enumerate(list_stocks):
        stock_i_area = np.sum(stock_i != -2)

        for j, stock_j in enumerate(list_stocks):
            if i == j:
                continue

            stock_j_area = np.sum(stock_j != -2)

            if stock_i_area < stock_j_area:
                return {"stock_idx": j, "size": (stock_i.shape[0], stock_i.shape[1]), "position": (0, 0)}

    return None


def combination_policy(observation, info):
    """
    Combination Algorithm - Kết hợp First-Fit, Best-Fit và Stock Merging.
    """
    action = first_fit_placement(observation)
    if action is None:
        action = best_fit_refinement(observation)
    if action is None:
        action = stock_merging(observation)
    if action is None:
        action = {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
    return action
