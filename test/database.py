from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Chuỗi kết nối đến MongoDB Atlas
uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"

# Tạo client và kết nối đến server MongoDB
client = MongoClient(uri, server_api=ServerApi('1'))

# 1. Hàm thêm dữ liệu


def add_user(collection, user_data):
    try:
        collection.insert_one(user_data)
        print(f"User {user_data['name']} đã được thêm vào.")
    except Exception as e:
        print(f"Không thể thêm dữ liệu: {e}")

    # 2. Hàm cập nhật dữ liệu


def update_user(collection, query, new_data):
    try:
        collection.update_one(query, {"$set": new_data})
        print(f"Dữ liệu đã được cập nhật cho user có query: {query}")
    except Exception as e:
        print(f"Không thể cập nhật dữ liệu: {e}")

    # 3. Hàm xóa dữ liệu


def delete_user(collection, query):
    try:
        result = collection.delete_one(query)
        if result.deleted_count > 0:
            print(f"User có query {query} đã bị xóa.")
        else:
            print(f"Không tìm thấy user với query {query} để xóa.")
    except Exception as e:
        print(f"Không thể xóa dữ liệu: {e}")


# Kiểm tra kết nối bằng lệnh ping
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")

    # Chọn cơ sở dữ liệu
    db = client["banking"]  # Tên cơ sở dữ liệu

    # Chọn collection
    collection = db["user_info"]  # Tên collection

    # Ví dụ sử dụng các hàm
    # Dữ liệu mẫu để thêm vào
    new_user = {
        "name": "Nguyen Van D",
        "age": 28,
        "phone": "0933445566",
        "address": "Huế, Việt Nam",
        "account_number": "789123456"
    }

    # Thêm user
    add_user(collection, new_user)

    # Cập nhật thông tin user
    update_user(collection, {"name": "Obama"}, {"age": 31})

    # Xóa user
    delete_user(collection, {"name": "Tran Thi B"})

except Exception as e:
    print(f"Kết nối thất bại: {e}")

# Đóng kết nối sau khi hoàn tất
client.close()
