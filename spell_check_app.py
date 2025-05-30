import streamlit as st
import pyperclip
from model import *

# Cấu hình trang
st.set_page_config(
    page_title="Check Spelling",
    page_icon="✏️",
    layout="wide"
)

# CSS để tùy chỉnh giao diện
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 50px;
        color: #333;
    }
    
    .text-area-container {
        margin-bottom: 30px;
    }
    
    .stTextArea > div > div > textarea {
        min-height: 300px;
        font-size: 16px;
        border: 3px solid #333;
        border-radius: 15px;
        padding: 15px;
    }
    
    .copy-button {
        margin-top: 10px;
    }
    
    .column-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown('<h1 class="main-title">CHECK SPELLING</h1>', unsafe_allow_html=True)

# Tạo 2 cột
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="column-header">Nhập văn bản:</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        "",
        height=300,
        key="input_text",
        placeholder="Nhập văn bản cần kiểm tra chính tả..."
    )

with col2:
    st.markdown('<div class="column-header">Sửa lỗi:</div>', unsafe_allow_html=True)
    
    # Xử lý kiểm tra chính tả đơn giản (có thể thay thế bằng thư viện chuyên dụng)
    corrected_text = call(input_text)  # Placeholder - có thể tích hợp spell checker thực tế
    
    corrected_display = st.text_area(
        "",
        value=corrected_text,
        height=300,
        key="corrected_text"
    )
    
    # Nút copy
    if st.button("📋 Copy văn bản đã kiểm tra", key="copy_button"):
        if corrected_display:
            try:
                pyperclip.copy(corrected_display)
                st.success("✅ Đã copy văn bản vào clipboard!")
            except:
                st.info("💡 Văn bản đã được chuẩn bị để copy. Bạn có thể chọn và copy thủ công.")
        else:
            st.warning("⚠️ Không có văn bản để copy!")

# Hướng dẫn sử dụng
st.markdown("---")
with st.expander("📖 Hướng dẫn sử dụng"):
    st.markdown("""
    1. **Nhập văn bản**: Dán hoặc gõ văn bản cần kiểm tra vào ô bên trái
    2. **Xem kết quả**: Văn bản đã được kiểm tra sẽ hiển thị ở ô bên phải
    3. **Copy kết quả**: Nhấn nút "Copy văn bản đã kiểm tra" để sao chép kết quả
    
    **Lưu ý**: Để có chức năng kiểm tra chính tả hoàn chỉnh, bạn có thể tích hợp:
    - `pyspellchecker` cho tiếng Anh
    - `vietspell` hoặc `underthesea` cho tiếng Việt
    - API của Google Translate hoặc các dịch vụ khác
    """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; margin-top: 50px;">Công cụ kiểm tra chính tả đơn giản</div>',
    unsafe_allow_html=True
)