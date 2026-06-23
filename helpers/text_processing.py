from underthesea import word_tokenize


def preprocess_vietnamese_text(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt bằng cách tách từ (word tokenization) sử dụng thư viện underthesea.

    Args:
        text (str): Văn bản tiếng Việt thô cần tách từ.

    Returns:
        str: Văn bản đã được tách từ (các từ ghép nối với nhau bằng dấu gạch dưới).
    """
    return word_tokenize(text, format="text")

