import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from helpers.constants import (PROMPT_GUARDRAIL_SYSTEM_PROMPT,
                               PROMPT_GUARDRAIL_USER_TEMPLATE)
from helpers.models import CustomOpenAIClient

logger = logging.getLogger(__name__)

_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

class PromptGuardrail:
    """
    Lớp bảo vệ đầu vào (Prompt Guardrail) để phát hiện và ngăn chặn các hành vi tấn công
    như Prompt Injection, Jailbreak, và rò rỉ prompt gốc (Prompt Leaking).
    """

    def __init__(self, llm_client: Optional[CustomOpenAIClient] = None):
        """
        Khởi tạo PromptGuardrail với cấu hình các mẫu Regex cơ bản và LLM client.

        Args:
            llm_client (Optional[CustomOpenAIClient]): Client LLM được dùng để kiểm tra ngữ nghĩa sâu.
        """
        self.llm_client = llm_client
        
        # Bổ sung và phân loại Pattern để dễ quản lý
        self.patterns: List[re.Pattern] = [
            # 1. Bỏ qua chỉ thị (Instruction Override)
            re.compile(r"ignore\s+(?:previous|above|the)\s+instructions", re.IGNORECASE),
            re.compile(r"bỏ\s+qua\s+(?:các\s+)?(?:chỉ\s+dẫn|chỉ\s+thị|quy\s+tắc)", re.IGNORECASE),
            re.compile(r"không\s+cần\s+tuân\s+theo", re.IGNORECASE),
            
            # 2. Rò rỉ Prompt (Prompt Leaking)
            re.compile(r"system\s+prompt|show\s+(?:your\s+)?prompt|reveal\s+(?:your\s+)?instructions", re.IGNORECASE),
            re.compile(r"cung\s+cấp\s+prompt\s+gốc|tiết\s+lộ\s+(?:chỉ\s+dẫn|prompt|quy\s+tắc)", re.IGNORECASE),
            re.compile(r"lặp\s+lại\s+câu\s+trên|in\s+ra\s+dòng\s+đầu\s+tiên", re.IGNORECASE),
            
            # 3. Jailbreak & Roleplay
            re.compile(r"jailbreak|dan\s+mode|developer\s+mode|dev\s+mode|do\s+anything\s+now", re.IGNORECASE),
            re.compile(r"bẻ\s+khóa|chế\s+độ\s+nhà\s+phát\s+triển", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+a|bạn\s+bây\s+giờ\s+là|hãy\s+đóng\s+vai", re.IGNORECASE),
            
            # 4. Kỹ thuật che giấu cơ bản (Mã hóa, script)
            re.compile(r"base64|hex", re.IGNORECASE),
            re.compile(r"<script>|os\.system|subprocess", re.IGNORECASE)
        ]

    def check_heuristics(self, query: str) -> Tuple[bool, str]:
        """
        Kiểm tra nhanh bằng Regex (Heuristics) để phát hiện các từ khóa hoặc mẫu câu tấn công phổ biến.

        Args:
            query (str): Câu hỏi/yêu cầu của người dùng.

        Returns:
            Tuple[bool, str]: Trả về một tuple gồm (is_injection, lý_do).
        """
        normalized_query = query.strip()
        for pattern in self.patterns:
            if pattern.search(normalized_query):
                return True, f"Bị chặn bởi bộ lọc từ khóa: Khớp với pattern '{pattern.pattern}'"
        return False, ""

    def check_llm(self, query: str) -> Tuple[bool, str]:
        """
        Sử dụng LLM để phân tích ngữ nghĩa sâu hơn về hành vi Prompt Injection hoặc Jailbreak.

        Args:
            query (str): Câu hỏi/yêu cầu của người dùng.

        Returns:
            Tuple[bool, str]: Trả về một tuple gồm (is_injection, lý_do).
        """
        if not self.llm_client:
            return False, "Không có LLM client để kiểm tra."
            
        system_prompt = PROMPT_GUARDRAIL_SYSTEM_PROMPT
        user_prompt = PROMPT_GUARDRAIL_USER_TEMPLATE.format(query=query)
        try:
            # Chú ý: Dùng await cho hàm gọi API bất đồng bộ
            response = self.llm_client.chat.completions.create(
                model=self.llm_client.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # Bắt buộc phải là 0 để model không sáng tạo
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            
            is_injection = bool(data.get("is_injection", False))
            reason = str(data.get("reason", "LLM Guardrail flagged this query."))
            
            return is_injection, reason
            
        except Exception as e:
            logger.error(f"Lỗi khi chạy LLM guardrail check: {e}")
            # Trong trường hợp lỗi API, nên cho qua (False) để không làm đứt trải nghiệm người dùng, 
            # hoặc chặn (True) nếu bạn muốn bảo mật cực đoan. Ở đây chọn cho qua.
            return False, f"Lỗi kiểm tra bảo mật: {str(e)}"

    def is_safe(self, query: str, check_llm: bool = False) -> Dict[str, Any]:
        """
        Luồng chính kiểm tra tính an toàn của truy vấn (kết hợp Heuristics và LLM).

        Args:
            query (str): Câu hỏi/yêu cầu của người dùng.
            check_llm (bool): Có sử dụng LLM để kiểm tra sâu hay không.

        Returns:
            Dict[str, Any]: Dict kết quả chứa trạng thái an toàn ("safe": bool) và lý do ("reason": str).
        """
        if not query or not query.strip():
            return {"safe": True, "reason": "Query rỗng"}
            
        # 1. Chạy chặn Heuristics (Nhanh, không tốn tiền)
        is_inj, reason = self.check_heuristics(query)
        if is_inj:
            logger.warning(f"[Guardrail] Đã chặn bằng Regex: {reason}")
            return {"safe": False, "reason": reason}
            
        # 2. Chạy chặn bằng LLM (Chậm, ngữ nghĩa)
        if check_llm and self.llm_client:
            is_inj, reason = self.check_llm(query)
            if is_inj:
                logger.warning(f"[Guardrail] Đã chặn bằng LLM: {reason}")
                return {"safe": False, "reason": reason}
                
        return {"safe": True, "reason": "An toàn"}


class GroundingGuardrail:
    """
    Lớp bảo vệ Grounding (Grounding Guardrail) để đối chiếu thông tin phản hồi của LLM
    với tài liệu gốc được truy vấn, nhằm tránh hiện tượng "ảo giác" (hallucination) 
    và trích dẫn sai luật.
    """

    def __init__(self):
        """
        Khởi tạo GroundingGuardrail.
        """

    def verify_citations(
        self, answer: str, retrieved_chunks: List[Any]
    ) -> Tuple[bool, List[int]]:
        """
        Phân tích phần 'Căn cứ pháp lý' trong câu trả lời của LLM, trích xuất cặp (số điều, số hiệu văn bản),
        sau đó tìm kiếm trong tất cả các chunk đã truy vấn để xác minh xem cặp thông tin này có tồn tại không.

        Args:
            answer (str): Câu trả lời từ LLM.
            retrieved_chunks (List[Any]): Danh sách các chunks tài liệu đã được truy vấn từ vector DB.

        Returns:
            Tuple[bool, List[int]]: Trả về tuple gồm (is_valid, matched_chunk_indices).
        """
        # Nếu trả lời từ chối do không đủ thông tin hoặc nằm ngoài phạm vi, bỏ qua kiểm tra
        if answer.strip() == "Không thể trả lời câu hỏi này":
            return True, []
        if any(phrase in answer.lower() for phrase in [
            "nằm ngoài phạm vi tư vấn", "không thích hợp", "vấn đề chính trị"
        ]):
            return True, []

        # 1. Phân tách phần câu hỏi chính và phần 'Căn cứ pháp lý'
        parts = re.split(r'(?i)\*?\\*?căn cứ pháp lý\*?\\*?:?', answer)
        if len(parts) < 2:
            logger.warning("LỖI KIỂM CHỨNG: Không tìm thấy phần 'Căn cứ pháp lý'.")
            return False, []

        basis_text = parts[1]
        # Regex tìm định dạng: "Điều X, ... - số_hiệu_văn_bản" ở cuối dòng
        llm_pattern = re.compile(r'(?i)điều\s+(\d+).*\s+[-\–\—\−\:|]\s*([a-zA-Z0-9/\-đĐ_]+)[.\,;\)*\s]*$')
        pairs = []
        for line in basis_text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = llm_pattern.search(line)
            if match:
                pairs.append((
                    int(match.group(1)),
                    match.group(2).strip().lower()
                ))
        if not pairs:
            logger.warning("LỖI KIỂM CHỨNG: Không có dòng căn cứ pháp lý nào đúng định dạng.")
            return False, []

        # 2. Tìm kiếm trong các chunks đã truy vấn để đối chiếu
        chunk_article_pattern = re.compile(r'(?i)điều\s+(\d+)')
        matched_indices = []

        for llm_article_int, llm_doc_num_str in pairs:
            found = False
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
                    found = True
                    matched_indices.append(idx)
                    break

            if not found:
                logger.warning(
                    f"LỖI ĐỐI CHIẾU: Không tìm thấy chunk khớp "
                    f"Điều {llm_article_int} - {llm_doc_num_str}"
                )
                return False, []

        logger.info("KẾT QUẢ KIỂM CHỨNG: Hợp lệ 100%!")
        print("KẾT QUẢ KIỂM CHỨNG: Hợp lệ 100%!")
        return True, matched_indices


class StreamLoopGuardrail:
    """
    Phát hiện vòng lặp token bị lặp đi lặp lại vô hạn trong quá trình LLM stream câu trả lời.
    Kết hợp hai phương pháp:
    1. Phát hiện dòng trùng lặp liên tiếp (nhanh, phát hiện lặp dòng rõ ràng).
    2. Tần suất chuỗi con trong cửa sổ trượt (phát hiện các kiểu lặp tinh vi hơn).
    """

    def __init__(
        self,
        max_dup_lines: int = 8,
        window_size: int = 600,
        min_repeat_len: int = 30,
        max_repeats: int = 6,
    ):
        """
        Khởi tạo StreamLoopGuardrail.

        Args:
            max_dup_lines (int): Số dòng trùng lặp liên tiếp tối đa cho phép.
            window_size (int): Kích thước cửa sổ trượt để kiểm tra lặp chuỗi con (số ký tự).
            min_repeat_len (int): Chiều dài tối thiểu của chuỗi con cần kiểm tra lặp.
            max_repeats (int): Số lần xuất hiện tối đa cho phép của chuỗi con trong cửa sổ.
        """
        self.max_dup_lines = max_dup_lines
        self.window_size = window_size
        self.min_repeat_len = min_repeat_len
        self.max_repeats = max_repeats

    def check(self, full_text: str) -> Tuple[bool, Optional[str]]:
        """
        Kiểm tra văn bản hiện tại xem có bị lặp token hay không.

        Args:
            full_text (str): Văn bản đầy đủ được tích lũy đến hiện tại.

        Returns:
            Tuple[bool, Optional[str]]: Trả về (is_looping, thông_báo_lỗi_hoặc_None).
        """

        # Tier 1: Kiểm tra các dòng trùng lặp liên tiếp
        lines = full_text.split("\n")
        if len(lines) >= self.max_dup_lines:
            last = lines[-1].strip()
            if last:
                dup_count = 0
                for line in reversed(lines):
                    if line.strip() == last:
                        dup_count += 1
                    else:
                        break
                if dup_count >= self.max_dup_lines:
                    return True, (
                        f"Phát hiện vòng lặp token: dòng '{last[:50]}...' "
                        f"lặp {dup_count} lần liên tiếp."
                    )

        # Tier 2: Sử dụng cửa sổ trượt để đếm tần suất chuỗi con
        if len(full_text) >= self.window_size:
            window = full_text[-self.window_size:]
            seen = {}
            for i in range(len(window) - self.min_repeat_len + 1):
                sub = window[i : i + self.min_repeat_len]
                seen[sub] = seen.get(sub, 0) + 1
                if seen[sub] >= self.max_repeats:
                    return True, (
                        f"Phát hiện vòng lặp token: chuỗi '{sub[:30]}...' "
                        f"lặp {seen[sub]} lần trong {self.window_size} ký tự cuối."
                    )

        return False, None


class LanguageGuardrail:
    """
    Phát hiện các token tiếng Trung/Nhật/Hàn (CJK) trong bộ tokenizer của LLM
    và tạo ra logit_bias để chặn việc sinh ra các ký tự này.
    """

    def __init__(self, tokenizer_path: str = ""):
        """
        Khởi tạo LanguageGuardrail.

        Args:
            tokenizer_path (str): Đường dẫn đến thư mục chứa tokenizer. 
                                  Nếu để trống, sẽ dùng đường dẫn mặc định trong workspace.
        """
        if not tokenizer_path:
            tokenizer_path = os.path.join(_data_dir, "gemma_4_tokenizer")
        self._tokenizer_path = tokenizer_path
        self._logit_bias: Optional[Dict[str, int]] = None

    def _is_cjk_ideogram(self, o: int) -> bool:
        """
        Kiểm tra xem mã unicode của ký tự có nằm trong các khối ký tự CJK hay không.

        Args:
            o (int): Mã Unicode của ký tự cần kiểm tra.

        Returns:
            bool: True nếu thuộc CJK, ngược lại False.
        """
        return (
            0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or
            0x20000 <= o <= 0x2A6DF or 0x2A700 <= o <= 0x2B739 or
            0x2B740 <= o <= 0x2B81D or 0x2B820 <= o <= 0x2CEAF or
            0x2CEB0 <= o <= 0x2EBEF or 0x30000 <= o <= 0x3134F or
            0x31350 <= o <= 0x323AF or 0xF900 <= o <= 0xFAFF or
            0x2F00 <= o <= 0x2FDF or
            0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or
            0xAC00 <= o <= 0xD7AF or 0x3000 <= o <= 0x303F
        )

    @property
    def logit_bias(self) -> Dict[str, int]:
        """
        Tạo hoặc lấy logit_bias dict để chặn các token CJK.

        Returns:
            Dict[str, int]: Logit bias dict với giá trị -100 cho các token CJK.
        """
        if self._logit_bias is not None:
            return self._logit_bias
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {self._tokenizer_path} for LanguageGuardrail...")
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, trust_remote_code=True)
        vocab_size = len(tokenizer)
        token_ids_nested = [[i] for i in range(vocab_size)]
        decoded_tokens = tokenizer.batch_decode(token_ids_nested, skip_special_tokens=False)
        logit_bias = {}
        for token_id, decoded_str in enumerate(decoded_tokens):
            clean_str = decoded_str.strip()
            for c in clean_str:
                if self._is_cjk_ideogram(ord(c)):
                    logit_bias[str(token_id)] = -100
                    break
            self._logit_bias = logit_bias
        logger.info(f"LanguageGuardrail: {len(logit_bias)} CJK tokens blocked")
        return self._logit_bias