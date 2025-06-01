# Vietnamese Spell Check App

Ứng dụng kiểm tra lỗi chính tả tiếng Việt sử dụng Python và TensorFlow.

## Yêu cầu hệ thống

- Python **3.11** trở lên
- pip
- CUDA (nếu dùng GPU)

## Hướng dẫn cài đặt

### 1. Cài đặt Python

Bạn có thể tải và cài đặt Python tại [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. Cài đặt TensorFlow

Cài đặt TensorFlow bằng pip:

```
pip install tensorflow
```

Nếu bạn sử dụng GPU, hãy đảm bảo đã cài đặt đúng CUDA và cuDNN tương thích. Hướng dẫn chi tiết tại: 👉 [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu)

3. Cài đặt các thư viện phụ thuộc
Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết:

```
pip install -r requirements.txt
```

4. Chạy ứng dụng với Streamlit
Sau khi cài đặt xong, chạy ứng dụng bằng lệnh sau:

```
streamlit run spell_check_app.py
```
Ứng dụng sẽ được mở trong trình duyệt mặc định tại địa chỉ: http://localhost:8501
