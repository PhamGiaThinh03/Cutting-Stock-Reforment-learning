################### ý tưởng ###############
# Duyệt qua từng sản phẩm theo thứ tự ban đầu.
# Tìm stock phù hợp nhất:
#     + Kiểm tra từng stock có thể chứa sản phẩm.
#     + Chọn stock để lại ít không gian dư nhất sau khi đặt sản phẩm.
# Nếu không có stock nào phù hợp, tạo một stock mới.
# Lặp lại cho đến khi tất cả sản phẩm được đặt xong.


import numpy as np

def best_fit_policy(observation, info):
    """
    Best-Fit Algorithm for 2D Cutting-Stock Problem.
    - Tìm stock để lại ít diện tích dư thừa nhất sau khi đặt sản phẩm.
    - Nếu không có stock phù hợp, mở một stock mới.
    """
    list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)  # Sắp xếp sản phẩm từ lớn đến nhỏ
    list_stocks = observation["stocks"]

    for prod in list_prods:
        if prod["quantity"] <= 0:
            continue  # Bỏ qua sản phẩm đã hết

        prod_w, prod_h = prod["size"]

        best_stock_idx, best_pos_x, best_pos_y = None, None, None
        min_remaining_area = float("inf")

        # Duyệt qua từng stock để tìm stock phù hợp nhất
        for idx, stock in enumerate(list_stocks):
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))

            if stock_w < prod_w or stock_h < prod_h:
                continue  # Bỏ qua nếu stock không đủ lớn

            # Tìm vị trí tối ưu để đặt sản phẩm
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                        # ✅ Tính diện tích trống còn lại sau khi đặt sản phẩm
                        remaining_area = np.count_nonzero(stock == -1) - (prod_w * prod_h)
                        if remaining_area < min_remaining_area:
                            best_stock_idx = idx
                            best_pos_x, best_pos_y = x, y
                            min_remaining_area = remaining_area

        # Nếu tìm thấy vị trí tối ưu trong stock hiện tại
        if best_stock_idx is not None:
            stock = list_stocks[best_stock_idx]
            stock[best_pos_x:best_pos_x + prod_w, best_pos_y:best_pos_y + prod_h] = 1  # Đánh dấu vùng đã sử dụng
            prod["quantity"] -= 1  # Giảm số lượng sản phẩm
            return {
                "stock_idx": best_stock_idx,
                "size": (prod_w, prod_h),
                "position": (best_pos_x, best_pos_y)
            }

    # Nếu không có stock nào phù hợp, mở stock mới
    new_stock_idx = len(list_stocks)
    return {
        "stock_idx": new_stock_idx,
        "size": (prod_w, prod_h),
        "position": (0, 0)
    }

