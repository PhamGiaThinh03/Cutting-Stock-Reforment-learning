
################## ý tưởng ############3#
# Sắp xếp danh sách sản phẩm từ lớn nhất đến nhỏ nhất dựa trên diện tích.
# Sắp xếp danh sách kho chứa từ lớn nhất đến nhỏ nhất dựa trên diện tích.
# Duyệt từng sản phẩm và thử đặt vào stock có sẵn.
# Nếu không tìm thấy chỗ phù hợp, mở một kho mới và đặt sản phẩm vào đó.
# Lặp lại đến khi tất cả sản phẩm được xử lý hoặc không còn chỗ để đặt.



import numpy as np



def first_fit_policy(observation, info):
    """
    First Fit Policy for 2D Cutting Stock Problem
    """
    list_prods = sorted(
        observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True
    )
    list_stocks = sorted(
        enumerate(observation["stocks"]),
        key=lambda x: np.sum(x[1] != -2),
        reverse=True
    )

    for prod in list_prods:
        if prod["quantity"] <= 0:
            continue

        prod_w, prod_h = prod["size"]
        stock_idx, pos_x, pos_y = None, None, None

        # Duyệt từng stock và tìm chỗ để đặt sản phẩm
        for idx, stock in list_stocks:
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))

            if stock_w < prod_w or stock_h < prod_h:
                continue  # Nếu kho không đủ lớn, bỏ qua

            # Tìm vị trí trống phù hợp trong stock
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                        stock_idx, pos_x, pos_y = idx, x, y
                        break
                if stock_idx is not None:
                    break

            if stock_idx is not None:
                break

        if stock_idx is None:
            # Nếu không tìm thấy kho phù hợp, tạo kho mới
            new_stock_idx = len(observation["stocks"])
            return {
                "stock_idx": new_stock_idx,
                "size": (prod_w, prod_h),
                "position": (0, 0)
            }

        return {
            "stock_idx": stock_idx,
            "size": (prod_w, prod_h),
            "position": (pos_x, pos_y)
        }

    return {
        "stock_idx": 0,
        "size": (0, 0),
        "position": (0, 0)
    }  # Trường hợp không có sản phẩm hợp lệ