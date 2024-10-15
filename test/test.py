import threading
import random
import time

# Luồng 1: Sinh số ngẫu nhiên từ 1 đến 10


def random_number_generator():
    while True:
        x = random.randint(1, 10)
        print(f"Random number: {x}")
        time.sleep(1)  # Dừng 1 giây để tránh chạy quá nhanh

# Luồng 2: Lấy số x + 2


def add_two(x):
    while True:
        result = x + 2
        print(f"{x} + 2 = {result}")
        time.sleep(1)  # Dừng 1 giây

# Luồng 3: Lấy số x * 3


def multiply_by_three(x):
    while True:
        result = x * 3
        print(f"{x} * 3 = {result}")
        time.sleep(1)  # Dừng 1 giây


# Biến x để chia sẻ giữa các luồng
x = random.randint(1, 10)

# Tạo và khởi chạy các luồng
thread1 = threading.Thread(target=random_number_generator)
thread2 = threading.Thread(target=add_two, args=(x,))
thread3 = threading.Thread(target=multiply_by_three, args=(x,))

thread1.start()
thread2.start()
thread3.start()

# Để các luồng chạy liên tục
thread1.join()
thread2.join()
thread3.join()
