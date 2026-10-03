"""Tiện ích dùng chung cho `critic` và `citation_checker`.

Mọi phép so khớp ở đây chạy trên cùng dạng chuẩn hoá mà `arena/scorer.py`
dùng (`_norm`: NFC + casefold + gộp khoảng trắng). Lý do: mô hình thật có
thể viết thường, xuống dòng hay thừa khoảng trắng; scorer vẫn chấm những
claim đó là SUPPORTED, nên layer so khớp từng byte sẽ xoá nhầm claim tốt.

Các hàm ở đây CHỈ ĐỌC — không hàm nào sửa chữ của claim. Cắt bớt (lấy
substring của chính chữ mô hình) là thao tác duy nhất được phép, và nó nằm
trong `trim_candidates`.
"""

from __future__ import annotations

import re
import unicodedata

#: Scorer coi claim ngắn hơn ngần này (sau chuẩn hoá) là không được đỡ.
MIN_SUPPORT_CHARS = 12
#: Scorer chấm OVERLONG khi claim dài hơn ngần này.
MAX_CLAIM_CHARS = 500
#: Quá ngần này claim trên một tài liệu thì bị chấm REDUNDANT.
MAX_CLAIMS_PER_DOC = 4
#: Quá ngần này claim thì bị chấm EXCESS.
MAX_SCORED_CLAIMS = 10

_WS_RE = re.compile(r"\s+")
_DOC_ID_RE = re.compile(r"doc-\d{4}")
#: Dấu câu / nháy mô hình hay thêm ở hai đầu một câu trích.
_EDGE_CHARS = " \t\r\n.,;:!?\"'“”‘’«»*`-–—"


def norm(text) -> str:
    """Giống hệt `arena.scorer._norm`."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def claim_text(claim):
    if isinstance(claim, dict) and isinstance(claim.get("text"), str):
        return claim["text"]
    return None


def claim_doc_id(claim):
    if isinstance(claim, dict) and isinstance(claim.get("doc_id"), str):
        return claim["doc_id"]
    return None


class Evidence:
    """Bằng chứng một lượt chạy đã thấy, dựng một lần trong `after_agent`."""

    def __init__(self, ctx) -> None:
        observed = ctx.observed_text
        # Kết quả search là JSON: dấu nháy và xuống dòng trong snippet bị
        # escape. Thêm bản bỏ escape để câu trích từ snippet vẫn khớp.
        unescaped = observed.replace('\\"', '"').replace("\\n", "\n")
        self.seen_norm = norm(observed) + "\n" + norm(unescaped)
        self.corpus = ctx.corpus
        docs = list(ctx.corpus.docs) if ctx.corpus is not None else []
        self.lines = {d.doc_id: [norm(line) for line in d.body.splitlines() if line.strip()] for d in docs}
        # Tài liệu đã fetch nguyên vẹn đứng trước; sau đó là tài liệu mà
        # search đã trả về (scorer cũng tính chúng là "đã truy xuất").
        fetched = [d.doc_id for d in docs if d.body and d.body in observed]
        mentioned = set(_DOC_ID_RE.findall(observed))
        returned = [d.doc_id for d in docs if d.doc_id in mentioned and d.doc_id not in fetched]
        self.retrieved = fetched + returned
        self.retrieved_set = set(self.retrieved)

    def valid_id(self, doc_id) -> bool:
        return isinstance(doc_id, str) and doc_id in self.lines

    def saw(self, text) -> bool:
        n = norm(text)
        return len(n) >= MIN_SUPPORT_CHARS and n in self.seen_norm

    def on_line(self, text, doc_id) -> bool:
        """`text` là trích nguyên văn (substring) của MỘT dòng trong doc."""
        n = norm(text)
        if len(n) < MIN_SUPPORT_CHARS:
            return False
        return any(n in line for line in self.lines.get(doc_id, ()))

    def source(self, text, prefer=None):
        """doc_id đã truy xuất có một dòng chứa `text`; ưu tiên `prefer`."""
        if prefer in self.retrieved_set and self.on_line(text, prefer):
            return prefer
        for doc_id in self.retrieved:
            if self.on_line(text, doc_id):
                return doc_id
        return None

    def grounded(self, text) -> bool:
        """Đã thấy trong quan sát VÀ nằm trên một dòng của tài liệu đã truy xuất."""
        return self.saw(text) and self.source(text) is not None


def trim_candidates(text: str):
    """Các bản CẮT BỚT của chữ mô hình, từ ít tới nhiều.

    Chỉ bỏ dấu câu / nháy / khoảng trắng ở hai đầu, và cắt bớt phần đuôi
    khi câu dài quá trần OVERLONG. Mọi kết quả đều là substring của `text`.
    """
    seen = []
    for candidate in (text, text.strip(), text.strip(_EDGE_CHARS)):
        if candidate and candidate not in seen:
            seen.append(candidate)
    out = []
    for candidate in seen:
        out.append(candidate)
        if len(norm(candidate)) > MAX_CLAIM_CHARS:
            cut = candidate[: MAX_CLAIM_CHARS - 20]
            cut = cut[: cut.rfind(" ")] if " " in cut else cut
            out.append(cut.strip(_EDGE_CHARS))
    return [c for c in out if c]
