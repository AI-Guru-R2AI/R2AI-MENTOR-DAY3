import json
import logging
import re
from typing import Any, Dict, Generator, List, Optional, Tuple

from fastembed import SparseTextEmbedding
from qdrant_client.http import models

from helpers import config
from helpers.decorators import retry
from helpers.models import CustomOpenAIClient, RAGQdrantClient
from helpers.text_processing import preprocess_vietnamese_text


class RAGPipeline:
    """
    Hệ thống pipeline chính cho RAG (Retrieval-Augmented Generation), quản lý kết nối cơ sở dữ liệu vector,
    các API Client LLM, các lớp Guardrails (bảo vệ đầu vào/đầu ra), mô hình sparse nhúng,
    và thực thi quy trình tìm kiếm và trả lời câu hỏi pháp lý.
    Lớp này sử dụng mẫu thiết kế Singleton.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Quản lý và trả về một instance duy nhất (Singleton) của lớp RAGPipeline.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Khởi tạo pipeline, thiết lập kết nối đến Qdrant DB và cấu hình các OpenAI client tùy chỉnh 
        cho embedding, reranker, reasoning LLM, non-reasoning LLM, và guardrails.
        """
        if getattr(self, "_initialized", False):
            return
        self.db_client = RAGQdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            collection_name=config.QDRANT_COLLECTION_NAME
        )
        self.embed_client = CustomOpenAIClient(
            api_key=config.EMBEDDING_OPENAI_API_KEY,
            base_url=config.EMBEDDING_OPENAI_BASE_URL,
            default_model=config.EMBEDDING_OPENAI_MODEL
        )
        self.rerank_client = CustomOpenAIClient(
            api_key=config.RERANKER_OPENAI_API_KEY,
            base_url=config.RERANKER_OPENAI_BASE_URL,
            default_model=config.RERANKER_OPENAI_MODEL
        )
        reasoning_llm_base_url = config.REASONING_BASE_URL
        if reasoning_llm_base_url:
            if not reasoning_llm_base_url.endswith("/v1") and not reasoning_llm_base_url.endswith("/v1/"):
                reasoning_llm_base_url = reasoning_llm_base_url.rstrip("/") + "/v1"
        self.reasoning_llm_client = CustomOpenAIClient(
            api_key=config.REASONING_API_KEY,
            base_url=reasoning_llm_base_url,
            default_model=config.REASONING_MODEL
        )
        non_reasoning_llm_base_url = config.NON_REASONING_BASE_URL
        if non_reasoning_llm_base_url:
            if not non_reasoning_llm_base_url.endswith("/v1") and not non_reasoning_llm_base_url.endswith("/v1/"):
                non_reasoning_llm_base_url = non_reasoning_llm_base_url.rstrip("/") + "/v1"
        self.non_reasoning_llm_client = CustomOpenAIClient(
            api_key=config.NON_REASONING_API_KEY,
            base_url=non_reasoning_llm_base_url,
            default_model=config.NON_REASONING_MODEL
        )
        guardrail_base_url = config.GUARDRAIL_OPENAI_BASE_URL
        if guardrail_base_url:
            if not guardrail_base_url.endswith("/v1") and not guardrail_base_url.endswith("/v1/"):
                guardrail_base_url = guardrail_base_url.rstrip("/") + "/v1"
        self.guardrail_client = CustomOpenAIClient(
            api_key=config.GUARDRAIL_OPENAI_API_KEY,
            base_url=guardrail_base_url,
            default_model=config.GUARDRAIL_OPENAI_MODEL
        )
        self._sparse_model = None
        self._prompt_guardrail = None
        self._grounding_guardrail = None
        self._stream_loop_guardrail = None
        self._language_guardrail = None
        self._initialized = True
        
    @property
    def prompt_guardrail(self):
        """
        Khởi tạo lười (lazy loading) và trả về bộ PromptGuardrail để kiểm tra câu hỏi đầu vào.
        """
        if getattr(self, "_prompt_guardrail", None) is None:
            from helpers.guardrails import PromptGuardrail
            self._prompt_guardrail = PromptGuardrail(llm_client=self.guardrail_client)
        return self._prompt_guardrail
        
    @property
    def grounding_guardrail(self):
        """
        Khởi tạo lười (lazy loading) và trả về bộ GroundingGuardrail để kiểm chứng nguồn trích dẫn pháp lý.
        """
        if getattr(self, "_grounding_guardrail", None) is None:
            from helpers.guardrails import GroundingGuardrail
            self._grounding_guardrail = GroundingGuardrail()
        return self._grounding_guardrail

    @property
    def stream_loop_guardrail(self):
        """
        Khởi tạo lười và trả về bộ StreamLoopGuardrail để theo dõi và phát hiện lặp token trong quá trình streaming.
        """
        if getattr(self, "_stream_loop_guardrail", None) is None:
            from helpers.guardrails import StreamLoopGuardrail
            self._stream_loop_guardrail = StreamLoopGuardrail()
        return self._stream_loop_guardrail

    @property
    def language_guardrail(self):
        """
        Khởi tạo lười và trả về bộ LanguageGuardrail để chặn các ký tự ngoại lai (tiếng Trung/Nhật/Hàn).
        """
        if getattr(self, "_language_guardrail", None) is None:
            from helpers.guardrails import LanguageGuardrail
            self._language_guardrail = LanguageGuardrail()
        return self._language_guardrail
        
    @property
    def sparse_model(self):
        """
        Khởi tạo lười mô hình SparseTextEmbedding dùng cho tìm kiếm từ khóa/BM25.
        """
        if self._sparse_model is None:
            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_model

        
    @retry(max_attempts=3)
    def retrieve_hybrid(self, query: str, limit: int = 5, query_filter: Optional[models.Filter] = None) -> List[Any]:
        """
        Thực hiện tìm kiếm lai (Hybrid Search) kết hợp vector dense và vector sparse.

        Args:
            query (str): Câu truy vấn của người dùng.
            limit (int): Số lượng kết quả tối đa cần lấy.
            query_filter (Optional[models.Filter]): Bộ lọc điều kiện tìm kiếm.

        Returns:
            List[Any]: Danh sách các điểm dữ liệu phù hợp thu được từ cơ sở dữ liệu.
        """
        # 1. Sinh dense vector truy vấn
        query_instruction = f"Instruct: Given a Vietnamese legal question, retrieve relevant legal passages that answer the question\nQuery: {query}"
        query_dense = self.embed_client.embeddings.create(
            input=[query_instruction],
            model=config.EMBEDDING_OPENAI_MODEL
        ).data[0].embedding
        
        # 2. Sinh sparse vector truy vấn
        query_segmented = preprocess_vietnamese_text(query)
        query_sparse_raw = list(self.sparse_model.query_embed(query_segmented))[0]
        query_sparse = models.SparseVector(
            indices=query_sparse_raw.indices.tolist(),
            values=query_sparse_raw.values.tolist()
        )
        
        # 3. DB retrieve
        return self.db_client.retrieve_hybrid(
            query_dense=query_dense,
            query_sparse=query_sparse,
            limit=limit,
            query_filter=query_filter
        )
        
    @retry(max_attempts=3)
    def retrieve_and_rerank(self, query: str, top_k: int = 3, candidate_limit: int = 20) -> List[Dict[str, Any]]:
        """
        Truy vấn các tài liệu thô bằng tìm kiếm lai, sau đó dùng mô hình Reranker để đánh giá lại điểm số.

        Args:
            query (str): Câu truy vấn của người dùng.
            top_k (int): Số lượng kết quả tốt nhất cần giữ lại sau khi rerank.
            candidate_limit (int): Số lượng tài liệu ứng viên cần lấy ra để chạy rerank.

        Returns:
            List[Dict[str, Any]]: Danh sách các kết quả đã được rerank và sắp xếp giảm dần theo điểm số.
        """
        candidates = self.retrieve_hybrid(query, limit=candidate_limit)
        if not candidates:
            return []
            
        pairs = [[query, p.payload.get("article_content", "")] for p in candidates]
        scores = self.rerank_client.rerank(query, [p.payload.get("article_content", "") for p in candidates])
        
        final_results = []
        for idx, score in enumerate(scores):
            final_results.append({
                "id": candidates[idx].id,
                "score": float(score),
                "payload": candidates[idx].payload
            })
            
        final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)
        return final_results[:top_k]

    def format_context_to_xml(self, chunks: List[Any]) -> str:
        """
        Chuyển đổi danh sách các chunks tài liệu thành cấu trúc văn bản định dạng XML.

        Args:
            chunks (List[Any]): Danh sách các chunks tài liệu.

        Returns:
            str: Chuỗi văn bản XML chứa thông tin chi tiết của các tài liệu.
        """
        xml_parts = ["<context>"]
        for idx, chunk in enumerate(chunks):
            payload = chunk.payload if hasattr(chunk, 'payload') else chunk.get('payload', {})
            doc_title = payload.get("doc_title", "Không rõ tiêu đề")
            doc_number = payload.get("doc_number", "Không rõ số hiệu")
            article_number = payload.get("article_number", "Không rõ số điều")
            article_content = payload.get("article_content", "")
            
            xml_parts.append(f'  <source id="{idx}">')
            xml_parts.append(f'    Văn bản: {doc_title}')
            xml_parts.append(f'    Số hiệu: {doc_number}')
            xml_parts.append(f'    Điều: {article_number}')
            xml_parts.append(f'    {article_content}')
            xml_parts.append('  </source>')
        xml_parts.append("</context>")
        return "\n".join(xml_parts)

    def format_context_with_limit(self, chunks: List[Any], reasoning: bool = True) -> Tuple[str, List[Any]]:
        """
        Cắt giảm độ dài ngữ cảnh ngữ nghĩa động (Dynamic Context Capping) để phù hợp với giới hạn context length của mô hình LLM.
        Sử dụng thuật toán tìm kiếm nhị phân để tìm vị trí cắt tối ưu đối với từng chunk nếu vượt giới hạn.

        Args:
            chunks (List[Any]): Danh sách các chunks tài liệu đã truy vấn.
            reasoning (bool): Sử dụng giới hạn của mô hình suy luận (reasoning) hay không.

        Returns:
            Tuple[str, List[Any]]: Trả về một tuple gồm chuỗi XML ngữ cảnh và danh sách các chunks được giữ lại (hoặc bị cắt ngắn).
        """
        import copy
        
        limit_env = config.REASONING_MODEL_CONTEXT if reasoning else config.NON_REASONING_MODEL_CONTEXT
        context_limit = limit_env // 2
        
        def count_tokens(text: str) -> int:
            return len(text.split())
        
        active_chunks = []
        previous_xml_parts = ["<context>"]
        
        for idx, chunk in enumerate(chunks):
            payload = chunk.payload if hasattr(chunk, 'payload') else chunk.get('payload', {})
            doc_title = payload.get("doc_title", "Không rõ tiêu đề")
            doc_number = payload.get("doc_number", "Không rõ số hiệu")
            article_number = payload.get("article_number", "Không rõ số điều")
            article_content = payload.get("article_content", "")
            
            words = article_content.split()
            low = 0
            high = len(words)
            best_w = -1
            
            while low <= high:
                mid = (low + high) // 2
                temp_content = " ".join(words[:mid])
                temp_block = [
                    f'  <source id="{idx}">',
                    f'    Văn bản: {doc_title}',
                    f'    Số hiệu: {doc_number}',
                    f'    Điều: {article_number}',
                    f'    {temp_content}',
                    f'  </source>'
                ]
                candidate_parts = previous_xml_parts + temp_block + ["</context>"]
                candidate_tokens = count_tokens("\n".join(candidate_parts))
                
                if candidate_tokens <= context_limit:
                    best_w = mid
                    low = mid + 1
                else:
                    high = mid - 1
                    
            if best_w == -1:
                # Không vừa nữa thì dừng
                break
                
            if best_w == len(words):
                # Vừa vặn toàn bộ chunk
                active_chunks.append(chunk)
                previous_xml_parts.extend([
                    f'  <source id="{idx}">',
                    f'    Văn bản: {doc_title}',
                    f'    Số hiệu: {doc_number}',
                    f'    Điều: {article_number}',
                    f'    {article_content}',
                    f'  </source>'
                ])
            else:
                # Bị cắt ngắn một phần
                truncated_content = " ".join(words[:best_w])
                chunk_copy = copy.deepcopy(chunk)
                if hasattr(chunk_copy, 'payload'):
                    chunk_copy.payload["article_content"] = truncated_content
                else:
                    chunk_copy["payload"]["article_content"] = truncated_content
                    
                active_chunks.append(chunk_copy)
                previous_xml_parts.extend([
                    f'  <source id="{idx}">',
                    f'    Văn bản: {doc_title}',
                    f'    Số hiệu: {doc_number}',
                    f'    Điều: {article_number}',
                    f'    {truncated_content}',
                    f'  </source>'
                ])
                break
                
        previous_xml_parts.append("</context>")
        context_xml = "\n".join(previous_xml_parts)
        return context_xml, active_chunks

    def append_inline_citations(self, answer: str, retrieved_chunks: List[Any]) -> Tuple[str, List[int]]:
        """
        Phân tích các căn cứ pháp lý trong phản hồi, đối chiếu và chèn thêm thẻ trích dẫn dạng inline [idx] 
        trong đó idx là chỉ số tương ứng của chunk tài liệu khớp.

        Args:
            answer (str): Câu trả lời từ LLM.
            retrieved_chunks (List[Any]): Danh sách các tài liệu đã truy vấn.

        Returns:
            Tuple[str, List[int]]: Trả về câu trả lời đã cập nhật trích dẫn và danh sách chỉ số các chunk khớp.
        """
        if answer.strip() == "Không thể trả lời câu hỏi này":
            return answer, []
        if any(phrase in answer.lower() for phrase in [
            "nằm ngoài phạm vi tư vấn", "không thích hợp", "vấn đề chính trị"
        ]):
            return answer, []

        parts = re.split(r'(?i)\*?\\*?căn cứ pháp lý\*?\\*?:?', answer)
        if len(parts) < 2:
            return answer, []

        main_text = parts[0]
        basis_text = parts[1]
        
        llm_pattern = re.compile(r'(?i)điều\s+(\d+).*\s+[-\–\—\−\:|]\s*([a-zA-Z0-9/\-đĐ_]+)[.\,;\)*\s]*$')
        chunk_article_pattern = re.compile(r'(?i)điều\s+(\d+)')
        
        new_basis_lines = []
        matched_indices = []
        
        for line in basis_text.splitlines():
            trimmed = line.strip()
            if not trimmed:
                new_basis_lines.append(line)
                continue
                
            match = llm_pattern.search(trimmed)
            if match:
                llm_article_int = int(match.group(1))
                llm_doc_num_str = match.group(2).strip().lower()
                
                # Tìm chunk tài liệu trùng khớp
                matched_chunk_idx = None
                for idx, chunk in enumerate(retrieved_chunks):
                    payload = chunk.payload if hasattr(chunk, 'payload') else chunk.get('payload', {})
                    real_doc_num_str = str(payload.get("doc_number", "")).strip().lower()
                    if llm_doc_num_str != real_doc_num_str:
                        continue
                    real_article_str = str(payload.get("article_number", ""))
                    chunk_match = chunk_article_pattern.search(real_article_str)
                    if not chunk_match:
                        continue
                    if llm_article_int == int(chunk_match.group(1)):
                        matched_chunk_idx = idx
                        break
                        
                if matched_chunk_idx is not None:
                    matched_indices.append(matched_chunk_idx)
                    new_basis_lines.append(line + f" [{matched_chunk_idx}]")
                else:
                    new_basis_lines.append(line)
            else:
                new_basis_lines.append(line)
                
        updated_basis = "\n".join(new_basis_lines)
        
        prefix_match = re.search(r'(?i)(\*?\\*?căn cứ pháp lý\*?\\*?:?)', answer)
        prefix = prefix_match.group(1) if prefix_match else "Căn cứ pháp lý:"
        
        updated_answer = main_text + prefix + updated_basis
        return updated_answer, matched_indices

    def retrieve_and_answer(
        self,
        query: str,
        reasoning: bool = True,
        top_k_retrieval: int = 5,
        top_k_rerank: int = 20,
        temperature: float = 0.0,
        top_k_sampling: int = 20,
        top_p: float = 1.0,
        min_p: float = 0.0,
        enable_reranker: bool = False,
        enable_prompt_guardrail: bool = False,
        enable_stream_loop_guardrail: bool = False,
        enable_grounding_guardrail: bool = False,
        enable_language_guardrail: bool = False
    ) -> Dict[str, Any]:
        """
        Luồng RAG đồng bộ chính: nhận câu hỏi, thực hiện chạy qua guardrails, 
        truy vấn tài liệu pháp lý, giới hạn ngữ cảnh, gọi LLM tạo phản hồi, 
        đối chiếu kiểm chứng thông tin và trả về kết quả cuối cùng.

        Args:
            query (str): Câu hỏi của người dùng.
            reasoning (bool): Sử dụng mô hình suy luận (reasoning) có khả năng thinking cao.
            top_k_retrieval (int): Số lượng tài liệu cần lấy.
            top_k_rerank (int): Số lượng tài liệu cần lấy ra chạy rerank (nếu bật).
            temperature (float): Nhiệt độ kiểm soát tính sáng tạo của LLM.
            top_k_sampling (int): Tham số lấy mẫu top_k của LLM.
            top_p (float): Tham số lấy mẫu top_p.
            min_p (float): Tham số lấy mẫu min_p.
            enable_reranker (bool): Có bật Reranker hay không.
            enable_prompt_guardrail (bool): Có bật kiểm soát Prompt Injection ở đầu vào.
            enable_stream_loop_guardrail (bool): Có bật kiểm soát lặp token.
            enable_grounding_guardrail (bool): Có bật đối chiếu căn cứ pháp lý chặt chẽ.
            enable_language_guardrail (bool): Có bật chặn các ký tự ngoại lai tiếng Trung/Nhật/Hàn.

        Returns:
            Dict[str, Any]: Dict chứa phản hồi ("answer"), trích dẫn ("citations"),
                            các chunks tài liệu đã dùng ("chunks"), trạng thái hợp lệ ("is_valid").
        """
        # 1. Prompt Guardrail Check
        if enable_prompt_guardrail:
            guardrail_result = self.prompt_guardrail.is_safe(query, check_llm=True)
            if not guardrail_result["safe"]:
                print(f"Blocked query due to prompt injection: {query} (Reason: {guardrail_result['reason']})")
                return {
                    "answer": "Yêu cầu không hợp lệ do nghi ngờ tấn công Prompt Injection.",
                    "citations": [],
                    "chunks": [],
                    "is_valid": False,
                    "raw_output": json.dumps({"is_injection": True, "reason": guardrail_result["reason"]})
                }

        # 2. Retrieval
        try:
            if enable_reranker:
                retrieved_chunks = self.retrieve_and_rerank(query, top_k=top_k_retrieval, candidate_limit=top_k_rerank)
            else:
                retrieved_chunks = self.retrieve_hybrid(query, limit=top_k_retrieval)
        except Exception as e:
            import traceback
            print(f"Retrieval error: {e}")
            traceback.print_exc()
            return {
                "answer": "Không thể truy vấn cơ sở dữ liệu pháp lý lúc này.",
                "citations": [],
                "chunks": [],
                "is_valid": False,
                "raw_output": f"Retrieval Error: {str(e)}"
            }

        # 3. Dynamic Context Capping and XML construction
        context_xml, active_chunks = self.format_context_with_limit(retrieved_chunks, reasoning=reasoning)

        # Định dạng chunks cho response
        serializable_chunks = []
        for chunk in active_chunks:
            payload = chunk.payload if hasattr(chunk, 'payload') else chunk.get('payload', {})
            serializable_chunks.append({
                "id": chunk.id if hasattr(chunk, 'id') else chunk.get('id'),
                "score": chunk.score if hasattr(chunk, 'score') else chunk.get('score', 0.0),
                "doc_title": payload.get("doc_title", "Không rõ tiêu đề"),
                "doc_number": payload.get("doc_number", "Không rõ số hiệu"),
                "article_number": payload.get("article_number", "Không rõ số điều"),
                "article_content": payload.get("article_content", "")
            })

        if not active_chunks:
            return {
                "answer": "Không thể trả lời câu hỏi này",
                "citations": [],
                "chunks": [],
                "is_valid": True,
                "raw_output": "{}"
            }

        # 3. Gọi LLM sinh văn bản phản hồi
        from helpers.constants import GROUNDED_SYSTEM_PROMPT as SYSTEM_PROMPT
        try:
            extra_kwargs = {}
            if enable_language_guardrail:
                extra_kwargs["logit_bias"] = self.language_guardrail.logit_bias
            user_content = context_xml + "\n" + query
            llm_client = self.reasoning_llm_client if reasoning else self.non_reasoning_llm_client
            response = llm_client.chat.completions.create(
                model=llm_client.default_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                extra_body={
                    "top_k": top_k_sampling,
                    "top_p": top_p,
                    "min_p": min_p,
                    "temperature": temperature,
                },
                **extra_kwargs
            )
            if getattr(response, 'choices', None) and response.choices:
                answer = response.choices[0].message.content.strip()
            else:
                raise ValueError("LLM response did not return any choices.")
            raw_output = answer
        except Exception as e:
            import logging
            logging.error(f"Failed to generate answer from LLM after retries: {e}")
            return {
                "answer": "Không thể trả lời câu hỏi này",
                "citations": [],
                "chunks": serializable_chunks,
                "is_valid": False,
                "raw_output": f"LLM Call Failure: {str(e)}"
            }

        # 4. Phân tích trích dẫn và chạy grounding guardrail kiểm chứng phản hồi
        citations = []
        if enable_grounding_guardrail:
            is_valid, citations_check = self.grounding_guardrail.verify_citations(answer, active_chunks)
            if not is_valid:
                answer = "Không thể trả lời câu hỏi này"
                citations = []
            else:
                answer, citations = self.append_inline_citations(answer, active_chunks)
        else:
            is_valid = True
            answer, citations = self.append_inline_citations(answer, active_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "chunks": serializable_chunks,
            "is_valid": is_valid,
            "raw_output": raw_output
        }

    def retrieve_and_answer_stream(
        self,
        query: str,
        reasoning: bool = True,
        top_k_retrieval: int = 5,
        top_k_rerank: int = 20,
        temperature: float = 0.0,
        top_k_sampling: int = 20,
        top_p: float = 1.0,
        min_p: float = 0.0,
        enable_reranker: bool = False,
        enable_prompt_guardrail: bool = False,
        enable_stream_loop_guardrail: bool = False,
        enable_grounding_guardrail: bool = False,
        enable_language_guardrail: bool = False,
    ) -> Generator[str, None, None]:
        """
        Luồng RAG bất đồng bộ / streaming: Trả về kết quả dưới dạng Generator để stream từng token (SSE).
        Đồng thời theo dõi kiểm tra lặp từ trong stream qua StreamLoopGuardrail.

        Args:
            (Các tham số tương tự hàm retrieve_and_answer)

        Yields:
            Generator[str, None, None]: Các dòng dữ liệu text/event-stream chứa JSON của token sinh ra.
        """
        try:
            # 1. Prompt Guardrail Check
            if enable_prompt_guardrail:
                guardrail_result = self.prompt_guardrail.is_safe(query, check_llm=True)
                if not guardrail_result["safe"]:
                    payload = {
                        "type": "error",
                        "answer": "Yêu cầu không hợp lệ do nghi ngờ tấn công Prompt Injection.",
                        "is_valid": False,
                        "raw_output": {"is_injection": True, "reason": guardrail_result["reason"]}
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    return

            # 2. Retrieval
            try:
                if enable_reranker:
                    retrieved_chunks = self.retrieve_and_rerank(query, top_k=top_k_retrieval, candidate_limit=top_k_rerank)
                else:
                    retrieved_chunks = self.retrieve_hybrid(query, limit=top_k_retrieval)
            except Exception as e:
                payload = {
                    "type": "error",
                    "answer": "Không thể truy vấn cơ sở dữ liệu pháp lý lúc này.",
                    "is_valid": False,
                    "raw_output": f"Retrieval Error: {str(e)}"
                }
                yield f"data: {json.dumps(payload)}\n\n"
                return

            # 3. Dynamic Context Capping and XML construction
            context_xml, active_chunks = self.format_context_with_limit(retrieved_chunks, reasoning=reasoning)

            serializable_chunks = []
            for chunk in active_chunks:
                payload = chunk.payload if hasattr(chunk, 'payload') else chunk.get('payload', {})
                serializable_chunks.append({
                    "id": chunk.id if hasattr(chunk, 'id') else chunk.get('id'),
                    "score": chunk.score if hasattr(chunk, 'score') else chunk.get('score', 0.0),
                    "doc_title": payload.get("doc_title", "Không rõ tiêu đề"),
                    "doc_number": payload.get("doc_number", "Không rõ số hiệu"),
                    "article_number": payload.get("article_number", "Không rõ số điều"),
                    "article_content": payload.get("article_content", "")
                })

            if not active_chunks:
                payload = {
                    "type": "done",
                    "answer": "Không thể trả lời câu hỏi này",
                    "citations": [],
                    "chunks": [],
                    "is_valid": True,
                    "raw_output": "{}"
                }
                yield f"data: {json.dumps(payload)}\n\n"
                return

            # 3. Stream LLM generation
            from helpers.constants import (GROUNDED_SYSTEM_PROMPT,
                                           GROUNDED_USER_TEMPLATE)
            full_text = ""
            answer_text = ""
            _last_checked = 0

            extra_kwargs = {}
            if enable_language_guardrail:
                extra_kwargs["logit_bias"] = self.language_guardrail.logit_bias

            try:
                llm_client = self.reasoning_llm_client if reasoning else self.non_reasoning_llm_client
                stream_resp = llm_client.chat.completions.create(
                    model=llm_client.default_model,
                    messages=[
                        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                        {"role": "user", "content": GROUNDED_USER_TEMPLATE.format(context_xml=context_xml, query=query)}
                    ],
                    stream=True,
                    extra_body={
                        "temperature": temperature,
                        "top_k": top_k_sampling,
                        "top_p": top_p,
                        "min_p": min_p,
                    },
                    **extra_kwargs
                )

                for chunk in stream_resp:
                    try:
                        choices = getattr(chunk, 'choices', None) or chunk.get('choices', None)
                        if not choices:
                            continue
                        first_choice = choices[0]
                        delta = first_choice.get('delta', None) if isinstance(first_choice, dict) else getattr(first_choice, 'delta', None)

                        if isinstance(delta, dict):
                            reasoning_content = delta.get('reasoning_content') or delta.get('reasoning') or delta.get('thinking') or delta.get('reason')
                            content = delta.get('content') or delta.get('text') or ''
                        else:
                            reasoning_content = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None) or getattr(delta, 'thinking', None) or getattr(delta, 'reason', None)
                            content = getattr(delta, 'content', None) or getattr(delta, 'text', None) or ''
                    except Exception:
                        reasoning_content = reasoning_content or None
                        content = content or ''

                    if reasoning_content:
                        full_text += reasoning_content
                        # Thường xuyên kiểm soát lặp từ trong suy luận
                        if enable_stream_loop_guardrail and len(full_text) - _last_checked >= 200:
                            _last_checked = len(full_text)
                            is_looping, msg = self.stream_loop_guardrail.check(full_text)
                            if is_looping:
                                logging.warning(f"[StreamLoop] {msg}")
                                payload = {"type": "invalidate", "message": "stream_loop_detected"}
                                yield f"data: {json.dumps(payload)}\n\n"
                                return
                        payload = {"type": "reasoning", "content": reasoning_content}
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif content:
                        full_text += content
                        answer_text += content
                        # Kiểm soát lặp từ trong câu trả lời chính thức
                        if enable_stream_loop_guardrail and len(full_text) - _last_checked >= 200:
                            _last_checked = len(full_text)
                            is_looping, msg = self.stream_loop_guardrail.check(full_text)
                            if is_looping:
                                logging.warning(f"[StreamLoop] {msg}")
                                payload = {"type": "invalidate", "message": "stream_loop_detected"}
                                yield f"data: {json.dumps(payload)}\n\n"
                                return
                        payload = {"type": "chunk", "content": content}
                        yield f"data: {json.dumps(payload)}\n\n"

            except Exception:
                # Dự phòng trong trường hợp streaming gặp ngoại lệ, gọi hàm tạo đồng bộ thay thế
                try:
                    response = llm_client.chat.completions.create(
                        model=llm_client.default_model,
                        messages=[
                            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                            {"role": "user", "content": GROUNDED_USER_TEMPLATE.format(context_xml=context_xml, query=query)}
                        ],
                        extra_body={
                            "temperature": temperature,
                            "top_k": top_k_sampling,
                            "top_p": top_p,
                            "min_p": min_p,
                        },
                        **extra_kwargs
                    )
                    if getattr(response, 'choices', None) and response.choices:
                        res_content = response.choices[0].message.content.strip()
                    else:
                        res_content = ""
                    full_text = res_content
                    answer_text = res_content
                    if full_text:
                        payload = {"type": "token", "text": full_text}
                        yield f"data: {json.dumps(payload)}\n\n"
                except Exception as e:
                    payload = {"type": "error", "answer": "LLM Call Failure: %s" % str(e)}
                    yield f"data: {json.dumps(payload)}\n\n"
                    return

        except Exception as e:
            import traceback
            traceback.print_exc()
            payload = {
                "type": "error",
                "answer": "Lỗi máy chủ nội bộ trong quá trình sinh câu trả lời.",
                "is_valid": False,
                "raw_output": str(e)
            }
            yield f"data: {json.dumps(payload)}\n\n"
            return

        # 4. Khi stream hoàn tất, chạy grounding guardrail để kiểm chứng và lập nguồn trích dẫn
        answer = answer_text
        raw_output = full_text

        citations = []
        is_valid = True

        if enable_grounding_guardrail:
            is_valid, citations_check = self.grounding_guardrail.verify_citations(answer, active_chunks)
            if not is_valid:
                payload = {"type": "invalidate", "message": "grounding_mismatch", "raw_output": full_text}
                yield f"data: {json.dumps(payload)}\n\n"
                return

        # Nối trích dẫn vào phía sau văn bản
        answer, citations = self.append_inline_citations(answer, active_chunks)

        # 5. Gửi sự kiện Done kèm metadata hoàn chỉnh
        payload = {
            "type": "done",
            "answer": answer,
            "citations": citations,
            "chunks": serializable_chunks,
            "is_valid": is_valid,
            "raw_output": raw_output
        }
        yield f"data: {json.dumps(payload)}\n\n"

