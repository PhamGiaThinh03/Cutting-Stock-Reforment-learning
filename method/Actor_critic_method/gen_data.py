import pandas as pd
import random

# Số batch cần tạo
num_batches = 20

# Kích thước tối đa của stock
max_width = 150
max_height = 150

# Hàm tạo danh sách stocks (cố định 10 stocks)
def generate_stocks():
    stocks = []
    for _ in range(10):  # Số lượng stocks cố định là 10
        width = random.randint(10, max_width)
        height = random.randint(10, max_height)
        stocks.append((width, height))
    return stocks

# Hàm tạo danh sách products (số lượng dao động từ 10 đến 20)
def generate_products(stocks):
    total_stock_area = sum(w * h for w, h in stocks)
    total_product_area = 0
    products = []

    num_products = random.randint(10, 40)  # Số lượng sản phẩm dao động từ 10 đến 20
    
    while len(products) < num_products and total_product_area < total_stock_area * 0.6:
        width = random.randint(15, 35)
        height = random.randint(15, 35)
        area = width * height

        if total_product_area + area <= total_stock_area:
            products.append((width, height))
            total_product_area += area
        else:
            break  # Dừng lại khi không thể thêm product nữa

    return products

# Tạo dữ liệu cho 10 batch
batch_data = []

for batch_id in range(1, num_batches + 1):
    stocks = generate_stocks()
    products = generate_products(stocks)
    
    for stock in stocks:
        batch_data.append((batch_id, "stock", stock[0], stock[1]))
    
    for product in products:
        batch_data.append((batch_id, "product", product[0], product[1]))

# Tạo DataFrame và lưu vào file CSV
df = pd.DataFrame(batch_data, columns=["batch_id", "type", "width", "height"])
csv_filename = "data/data.csv"
df.to_csv(csv_filename, index=False)