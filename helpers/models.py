from typing import Any, List, Optional

import requests
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models


class RAGQdrantClient:
    """
    Client kết nối Qdrant phục vụ cho hệ thống RAG. 
    Lớp này được thiết kế theo mẫu Singleton để tránh tạo nhiều kết nối trùng lặp.
    """
    _instances = {}

    def __new__(cls, host: str, port: int, collection_name: str):
        """
        Quản lý việc khởi tạo instance duy nhất dựa trên bộ tham số (host, port, collection_name).
        """
        key = (host, port, collection_name)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            instance._initialized = False
        return cls._instances[key]

    def __init__(self, host: str, port: int, collection_name: str):
        """
        Khởi tạo kết nối QdrantClient nếu chưa được khởi tạo trước đó.

        Args:
            host (str): Địa chỉ máy chủ Qdrant.
            port (int): Cổng dịch vụ Qdrant.
            collection_name (str): Tên collection dữ liệu pháp lý cần truy vấn.
        """
        if getattr(self, "_initialized", False):
            return
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self._initialized = True
        
    def collection_exists(self) -> bool:
        """
        Kiểm tra sự tồn tại của collection trên Qdrant.

        Returns:
            bool: True nếu collection đã tồn tại, ngược lại False.
        """
        return self.client.collection_exists(collection_name=self.collection_name)
        
    def delete_collection(self):
        """
        Xóa collection hiện tại khỏi Qdrant nếu nó tồn tại.
        """
        if self.collection_exists():
            self.client.delete_collection(collection_name=self.collection_name)
            
    def create_collection(self, dense_size: int = 1024):
        """
        Tạo mới một collection với cấu hình vector hỗ trợ cả tìm kiếm Dense (vector dày đặc)
        và Sparse (vector thưa thớt).

        Args:
            dense_size (int): Kích thước chiều của dense vector (mặc định là 1024).
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            }
        )
        
    def upsert_points(self, points: List[models.PointStruct]):
        """
        Đưa thêm dữ liệu (chèn/cập nhật các points) vào collection.

        Args:
            points (List[models.PointStruct]): Danh sách điểm dữ liệu cần lưu trữ.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def retrieve_hybrid(self, query_dense: List[float], query_sparse: models.SparseVector, limit: int = 5, query_filter: Optional[models.Filter] = None) -> List[Any]:
        """
        Thực hiện truy vấn lai (Hybrid Search) kết hợp kết quả tìm kiếm Dense (Dense Search)
        và Sparse (Sparse Search) sử dụng thuật toán Reciprocal Rank Fusion (RRF).

        Args:
            query_dense (List[float]): Vector đặc trưng của truy vấn (dense).
            query_sparse (models.SparseVector): Vector thưa của truy vấn (sparse).
            limit (int): Số lượng kết quả tối đa cần trả về.
            query_filter (Optional[models.Filter]): Bộ lọc điều kiện tìm kiếm.

        Returns:
            List[Any]: Danh sách các điểm dữ liệu phù hợp nhất kèm payload.
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=20),
                models.Prefetch(query=query_sparse, using="sparse", limit=20)
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=query_filter,
            limit=limit
        )
        return results.points



class CustomOpenAIClient(OpenAI):
    """
    Client OpenAI tùy chỉnh kế thừa từ OpenAI.
    Hỗ trợ mẫu Singleton và bổ sung các hàm chuyên dụng như kết nối API Reranker.
    """
    _instances = {}

    def __new__(cls, api_key: str = None, base_url: str = None, default_model: str = None, **kwargs):
        """
        Quản lý việc tạo instance duy nhất dựa trên bộ tham số (api_key, base_url, default_model).
        """
        key = (api_key, base_url, default_model)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            instance._initialized = False
        return cls._instances[key]

    def __init__(self, api_key: str = None, base_url: str = None, default_model: str = None, **kwargs):
        """
        Khởi tạo CustomOpenAIClient.

        Args:
            api_key (str): Khóa API để xác thực.
            base_url (str): Địa chỉ API Endpoint.
            default_model (str): Tên mô hình mặc định sẽ sử dụng.
        """
        if getattr(self, "_initialized", False):
            return
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.default_model = default_model
        self._initialized = True
        
    def rerank(self, query: str, documents: List[str], model: str = None) -> List[float]:
        """
        Gọi API Reranker để đánh giá lại điểm độ tương đồng giữa câu truy vấn và danh sách tài liệu.

        Args:
            query (str): Câu hỏi/yêu cầu từ người dùng.
            documents (List[str]): Danh sách nội dung các tài liệu cần đánh giá điểm.
            model (str): Mô hình rerank sử dụng (nếu có, nếu không sẽ dùng default_model).

        Returns:
            List[float]: Danh sách điểm số độ tương đồng tương ứng với từng tài liệu.
        """
        model_name = model or self.default_model
        url = str(self.base_url).rstrip("/")
        # Tự động hiệu chỉnh endpoint rerank dựa trên base_url
        if not url.endswith("/rerank") and not url.endswith("/v1/rerank"):
            if url.endswith("/v1"):
                url = url + "/rerank"
            else:
                url = url + "/v1/rerank"
                
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key and self.api_key != "not_needed":
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "model": model_name,
            "query": query,
            "documents": documents
        }
        
        try:
            # Thử gửi request theo định dạng chuẩn
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code != 200:
                # Nếu thất bại, thử định dạng TEI (Text Embeddings Inference)
                payload_tei = {
                    "model": model_name,
                    "query": query,
                    "texts": documents
                }
                response = requests.post(url, headers=headers, json=payload_tei, timeout=15)
        except Exception:
            # Nếu gặp lỗi kết nối, thử chuyển đổi giữa endpoint /rerank và /v1/rerank
            if "/v1/rerank" in url:
                alt_url = url.replace("/v1/rerank", "/rerank")
            else:
                alt_url = url.replace("/rerank", "/v1/rerank")
            try:
                response = requests.post(alt_url, headers=headers, json=payload, timeout=15)
                if response.status_code != 200:
                    payload_tei = {
                        "model": model_name,
                        "query": query,
                        "texts": documents
                    }
                    response = requests.post(alt_url, headers=headers, json=payload_tei, timeout=15)
            except Exception as e:
                raise Exception(f"Rerank API connection failed to both endpoints: {e}")
                
        if response.status_code != 200:
            raise Exception(f"Rerank API request failed with status {response.status_code}: {response.text}")
            
        res_data = response.json()
        scores = [0.0] * len(documents)
        results_list = []
        if isinstance(res_data, list):
            results_list = res_data
        elif isinstance(res_data, dict):
            results_list = res_data.get("results", res_data.get("data", []))
            
        for item in results_list:
            idx = item.get("index")
            score = item.get("score", item.get("relevance_score", 0.0))
            if idx is not None and 0 <= idx < len(scores):
                scores[idx] = score
        return scores

