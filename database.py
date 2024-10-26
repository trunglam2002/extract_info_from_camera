from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://nguyentrunglam2002:lampro3006@userdata.av8zp.mongodb.net/?retryWrites=true&w=majority&appName=UserData"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["banking"]
collection = db["user_info"]


def query_user_info(name):
    user_info = collection.find_one({"name": name})
    if user_info:
        return user_info
    else:
        return None
