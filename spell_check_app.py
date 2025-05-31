from collections import Counter
from keras.models import load_model
from nltk import ngrams,word_tokenize
import streamlit as st
import pyperclip
import numpy as np
import re
import string
import difflib

model = load_model('./spelling.h5')

NGRAM=2
MAXLEN=40
alphabet = ['\x00', ' ', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ', 'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ', 'đ', 'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Â', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ă', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ô', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ơ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ', 'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ê', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ư', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị', 'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ', 'Đ']
letters=list("abcdefghijklmnopqrstuvwxyzáàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđABCDEFGHIJKLMNOPQRSTUVWXYZÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÉÈẺẼẸÊẾỀỂỄỆÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴĐ")
accepted_char=list((string.digits + ''.join(letters)))

def call(sentence):
    def encoder_data(text, maxlen=MAXLEN):
            text = "\x00" + text
            x = np.zeros((maxlen, len(alphabet)))
            for i, c in enumerate(text[:maxlen]):
                x[i, alphabet.index(c)] = 1
            if i < maxlen - 1:
              for j in range(i+1, maxlen):
                x[j, 0] = 1
            return x

    def decoder_data(x):
        x = x.argmax(axis=-1)
        return ''.join(alphabet[i] for i in x)

    def nltk_ngrams(words, n=2):
        return ngrams(words.split(), n)

    def guess(ngram):
        text = ' '.join(ngram)
        preds = model.predict(np.array([encoder_data(text)]), verbose=0)
        return decoder_data(preds[0]).strip('\x00')

    def correct(sentence):
        for i in sentence:
            if i not in accepted_char:
                sentence=sentence.replace(i," ")
        ngrams = list(nltk_ngrams(sentence, n=NGRAM))
        guessed_ngrams = list(guess(ngram) for ngram in ngrams)

        print("N gram", ngrams)
        print("guess", guessed_ngrams)

        candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
        for nid, ngram in (enumerate(guessed_ngrams)):
            for wid, word in (enumerate(re.split(' +', ngram))):
                candidates[nid + wid].update([word])

        output = ' '.join(c.most_common(1)[0][0] for c in candidates)
        return output

    guess = correct(sentence)

    return guess

def highlight_differences(original, corrected):
    """
    So sánh văn bản gốc và văn bản đã sửa, trả về HTML với từ được sửa được đánh dấu
    """
    if not original.strip() or not corrected.strip():
        return corrected
    
    # Tách từ để so sánh
    original_words = original.split()
    corrected_words = corrected.split()
    
    # Sử dung difflib để tìm sự khác biệt
    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    
    highlighted_html = ""
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Từ giống nhau
            highlighted_html += ' '.join(corrected_words[j1:j2]) + ' '
        elif tag == 'replace':
            # Từ được thay thế (sửa lỗi)
            original_part = ' '.join(original_words[i1:i2])
            corrected_part = ' '.join(corrected_words[j1:j2])
            highlighted_html += f'<span style="background-color: rgb(144 150 164);padding: 2px 4px; border-radius: 3px; margin: 1px;text-decoration:underline;" title="Từ gốc: {original_part}">{corrected_part}</span> '
    return highlighted_html.strip()

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
    
    .column-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }
    
    .highlighted-text {
        font-size: 16px;
        line-height: 1.6;
        padding: 15px;
        border: 3px solid #333;
        border-radius: 15px;
        background-color: #262730;
        min-height: 300px;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown('<h1 class="main-title">CHECK SPELLING</h1>', unsafe_allow_html=True)
st.markdown('<div class="column-header">Nhập văn bản:</div>', unsafe_allow_html=True)
input_text = st.text_area(
    "",
    key="input_text",
    placeholder="Nhập văn bản cần kiểm tra chính tả..."
)

st.markdown('<div class="column-header">Sửa lỗi:</div>', unsafe_allow_html=True)

# Xử lý kiểm tra chính tả bằng hàm call()
corrected_text = ""
if input_text.strip():  # Chỉ xử lý khi có văn bản đầu vào
    try:
        corrected_text = call(input_text)
    except Exception as e:
        st.error(f"Lỗi khi xử lý văn bản: {str(e)}")
        corrected_text = input_text  # Fallback về văn bản gốc nếu có lỗi

# Hiển thị văn bản với highlighting
if corrected_text:
    highlighted_text = highlight_differences(input_text, corrected_text)
    st.markdown(
        f'<div class="highlighted-text">{highlighted_text}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="highlighted-text">Nhập văn bản bên trên để xem kết quả sửa lỗi...</div>',
        unsafe_allow_html=True
    )


# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; margin-top: 50px;">Công cụ kiểm tra chính tả đơn giản</div>',
    unsafe_allow_html=True
)
