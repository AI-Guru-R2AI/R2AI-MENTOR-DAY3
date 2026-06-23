# `Legal RAG Chatbot (r2ai)

Hệ thống **RAG (Retrieval-Augmented Generation)** chuyên biệt để tư vấn và giải đáp thắc mắc liên quan đến **Luật Doanh nghiệp Việt Nam**. Hệ thống kết hợp cơ chế tìm kiếm lai (Hybrid Search), chấm điểm lại (Reranking), và bộ năm lớp lá chắn bảo mật (Guardrails) nhằm tối ưu hóa câu trả lời và loại bỏ triệt để hiện tượng ảo giác (hallucination).

---

## 1. Tính năng nổi bật
* **Tìm kiếm lai (Hybrid Search)**: Kết hợp giữa Dense Search (truy vấn ngữ nghĩa sâu) và Sparse Search (truy vấn từ khóa/BM25 thông qua thư viện FastEmbed) trên cơ sở dữ liệu vector Qdrant.
* **Đánh giá lại (Reranking)**: Sử dụng mô hình Reranker (như BGE-Reranker) thông qua API OpenAI tùy chỉnh để nâng cao độ chính xác của các tài liệu tìm thấy.
* **Bộ Guardrails toàn diện**:
  * **Prompt Guardrail**: Sử dụng Regex và LLM phân tích ngữ nghĩa sâu để chặn đứng các cuộc tấn công Prompt Injection, Jailbreak, và rò rỉ prompt.
  * **Grounding Guardrail**: Tự động phân tích các dòng căn cứ pháp lý trong câu trả lời của LLM và đối chiếu chặt chẽ 100% với các chunks tài liệu gốc được truy vấn.
  * **Stream Loop Guardrail**: Theo dõi và tự động cắt luồng stream khi phát hiện vòng lặp vô hạn của token.
  * **Language Guardrail**: Áp dụng logit_bias để lọc bỏ hoàn toàn các ký tự ngoại lai (như CJK - tiếng Trung/Nhật/Hàn) trong lúc sinh từ đối với mô hình Gemma.
* **Hỗ trợ Streaming**: Giao tiếp thời gian thực qua Server-Sent Events (SSE).

---

## 2. Cấu trúc dự án

Dự án được phân bổ thành các thư mục và file chính như sau:

```
r2ai/
├── app/
│   ├── server.py              # FastAPI server (quản lý endpoints API và serve frontend tĩnh)
│   └── static/                # Giao diện người dùng tĩnh (HTML, CSS, JS)
├── helpers/
│   ├── __init__.py
│   ├── config.py              # Quản lý cấu hình và biến môi trường (.env)
│   ├── constants.py           # Quản lý các system/user prompts và prompt guardrails templates
│   ├── decorators.py          # Bộ decorator tự động retry khi gặp lỗi kết nối API
│   ├── guardrails.py          # Bộ điều hướng an toàn (Prompt, Grounding, StreamLoop, Language)
│   ├── models.py              # Client tùy chỉnh cho Qdrant DB và Custom OpenAI (LLM, Reranker)
│   ├── pipeline.py            # RAG Pipeline cốt lõi kết hợp retrieval, reranking, capping và LLM
│   └── text_processing.py     # Hỗ trợ tách từ tiếng Việt (Word Segmentation) bằng underthesea
├── data/                      # Lưu trữ bộ Tokenizer và dữ liệu bổ trợ
├── Dockerfile                 # Dockerfile build image cho ứng dụng FastAPI
├── docker-compose.yml         # Cấu hình container chạy FastAPI và Qdrant DB song song
├── requirements.txt           # Danh sách các thư viện Python phụ thuộc
└── README.md                  # Tài liệu hướng dẫn sử dụng dự án
```

---

## 3. Cấu hình biến môi trường (`.env`)

Tạo file `.env` từ file mẫu cấu hình:
```bash
cp .env.example .env
```
Sau đó, hãy mở `.env` lên và điền các thông số API Keys và kết nối của bạn.

---

## 4. Hướng dẫn chạy và triển khai (Deployment)

### Cách 1: Triển khai nhanh bằng Docker Compose (Khuyên dùng)

Cách thức này sẽ tự động khởi chạy và liên kết dịch vụ **FastAPI Server** cùng cơ sở dữ liệu **Qdrant DB**.

1. Đảm bảo đã có file `.env` (với cấu hình kết nối `QDRANT_HOST=qdrant` hoặc phù hợp với mạng Docker).
2. Chạy lệnh:
   ```bash
   docker-compose up --build -d
   ```
3. Truy cập ứng dụng:
   * **Giao diện Web Chatbot**: [http://localhost:8000/](http://localhost:8000/)
   * **Tài liệu API Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Cách 2: Triển khai thủ công (Sử dụng `uv` và Python 3.11)

Yêu cầu máy đã cài sẵn công cụ quản lý package [uv](https://github.com/astral-sh/uv) và Python 3.11, đồng thời có một instance Qdrant DB đang hoạt động.

1. **Khởi tạo môi trường ảo Python 3.11 bằng `uv`**:
   ```bash
   uv venv --python 3.11
   ```

2. **Kích hoạt môi trường ảo**:
   * Trên macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   * Trên Windows:
     ```cmd
     .venv\Scripts\activate
     ```

3. **Cài đặt các thư viện phụ thuộc cực nhanh bằng `uv`**:
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Cấu hình file `.env`**:
   Đảm bảo cấu hình biến `QDRANT_HOST` trỏ đến địa chỉ Qdrant DB đang hoạt động (ví dụ: `localhost`).

5. **Khởi chạy ứng dụng FastAPI**:
   ```bash
   uv run uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
   ```

6. Mở trình duyệt và truy cập [http://localhost:8000/](http://localhost:8000/).

