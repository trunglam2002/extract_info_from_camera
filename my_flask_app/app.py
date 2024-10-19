from flask import Flask, render_template, request, redirect, url_for
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId

app = Flask(__name__)

# Kết nối đến MongoDB Atlas
uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]  # Chọn cơ sở dữ liệu
collection = db["user_info"]  # Chọn collection

# Trang chủ với form để thêm người dùng


@app.route('/')
def index():
    users = collection.find()  # Lấy tất cả user từ MongoDB
    return render_template('index.html', users=users)

# Hàm xử lý thêm người dùng


@app.route('/add_user', methods=['POST'])
def add_user():
    name = request.form.get('name')
    age = request.form.get('age')
    phone = request.form.get('phone')
    address = request.form.get('address')
    account_number = request.form.get('account_number')

    # Thêm dữ liệu vào MongoDB
    user_data = {
        "name": name,
        "age": int(age),
        "phone": phone,
        "address": address,
        "account_number": account_number
    }
    collection.insert_one(user_data)
    return redirect(url_for('index'))

# Hàm xóa người dùng


@app.route('/delete_user/<user_id>')
def delete_user(user_id):
    collection.delete_one({"_id": ObjectId(user_id)})
    return redirect(url_for('index'))

# Hàm hiển thị form chỉnh sửa người dùng


@app.route('/edit_user/<user_id>')
def edit_user(user_id):
    # Lấy thông tin user theo id
    user = collection.find_one({"_id": ObjectId(user_id)})
    return render_template('edit_user.html', user=user)

# Hàm xử lý chỉnh sửa người dùng


@app.route('/update_user/<user_id>', methods=['POST'])
def update_user(user_id):
    name = request.form.get('name')
    age = request.form.get('age')
    phone = request.form.get('phone')
    address = request.form.get('address')
    account_number = request.form.get('account_number')

    # Cập nhật dữ liệu người dùng
    collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {
            "name": name,
            "age": int(age),
            "phone": phone,
            "address": address,
            "account_number": account_number
        }}
    )
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
