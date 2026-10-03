"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, trần recall là 0.5 với MỌI harness đúng luật, vì đúng lý do đó —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re

from harness.layers.evidence import (
    MAX_CLAIMS_PER_DOC,
    MAX_SCORED_CLAIMS,
    MIN_SUPPORT_CHARS,
    Evidence,
    claim_doc_id,
    claim_text,
    norm,
    trim_candidates,
)
from harness.middleware import Middleware


#: Phương án đánh chữ trong câu hỏi tổng hợp: "(a) ...; (b) ...; (c) ...".
_OPTION_RE = re.compile(r"\(([a-eA-E])\)\s*(.+?)\s*(?=[;,]?\s*(?:hoặc\s+)?\([a-eA-E]\)|;\s*$|\.(?:\s|$)|$)", re.S)
#: verdict chỉ ghi chữ cái: "b", "(b)", "b)", "phương án b".
_LETTER_RE = re.compile(r"^\W*(?:phương án|đáp án|option)?\W*([a-eA-E])\W*$")


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    #: Liên từ mô hình dùng để dán hai nửa câu của hai tài liệu.
    JOINERS = (" và ", "; ", " nhưng ", ", trong khi ", " còn ")

    def _ground(self, ev, claim, text):
        """Bản giữ được của claim (nguyên văn hoặc cắt bớt), hoặc None.

        Không đổi doc_id: gắn lại nguồn là việc của `citation_checker`
        (chạy TRƯỚC critic). Ở đây chỉ hỏi "có phải bịa không".
        """
        for candidate in trim_candidates(text):
            if ev.grounded(candidate):
                return claim if candidate == text else {**claim, "text": candidate}
        return None

    def _split(self, ev, text):
        """Tách câu ghép tại chỗ dán; hai nửa phải thuộc hai tài liệu khác nhau."""
        for joiner in self.JOINERS:
            start = text.find(joiner)
            while start != -1:
                halves = []
                for part in (text[:start], text[start + len(joiner):]):
                    grounded = self._ground(ev, {"text": part}, part)
                    if grounded is None:
                        break
                    # Nửa câu mới tách chưa có nguồn: gắn tài liệu thật sự chứa nó.
                    halves.append({"text": grounded["text"], "doc_id": ev.source(grounded["text"])})
                if len(halves) == 2 and halves[0]["doc_id"] != halves[1]["doc_id"]:
                    return halves
                start = text.find(joiner, start + 1)
        return None

    @staticmethod
    def _cap(claims):
        """Bỏ claim trùng và claim vượt trần chấm điểm (REDUNDANT / EXCESS)."""
        kept, texts, per_doc = [], [], {}
        for claim in claims:
            n = norm(claim["text"])
            doc_id = claim_doc_id(claim)
            if any(n in t for t in texts):
                continue  # trùng (hoặc nằm trong) một claim đã giữ
            if per_doc.get(doc_id, 0) >= MAX_CLAIMS_PER_DOC or len(kept) >= MAX_SCORED_CLAIMS:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            texts.append(n)
            kept.append(claim)
        return kept

    @staticmethod
    def _settle_verdict(ctx, report):
        """Brief tổng hợp: giúp verdict nêu ĐÚNG MỘT phương án — không tự chọn hộ.

        Chỉ dùng lựa chọn mô hình đã tự viết ra: verdict chỉ có chữ cái thì
        đổi sang câu phương án; verdict rỗng/mơ hồ mà answer nêu đúng một
        phương án thì chép phương án đó sang verdict.
        """
        options = {m.group(1).lower(): m.group(2).strip(" .;,") for m in _OPTION_RE.finditer(ctx.question)}
        options = {k: v for k, v in options.items() if len(v) >= MIN_SUPPORT_CHARS}
        if len(options) < 2:
            return
        verdict = report.get("verdict")
        verdict = verdict if isinstance(verdict, str) else ""
        letter = _LETTER_RE.match(verdict)
        if letter and letter.group(1).lower() in options:
            report["verdict"] = options[letter.group(1).lower()]
            return
        def asserted(text):
            return [k for k, phrase in options.items() if norm(phrase) in norm(text)]
        if verdict.strip() and len(asserted(verdict)) == 1:
            return
        answer = report.get("answer") if isinstance(report.get("answer"), str) else ""
        in_answer = asserted(answer)
        if len(in_answer) == 1:
            report["verdict"] = options[in_answer[0]]

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if isinstance(claims, list) and claims and ctx.corpus is not None:
            ev = Evidence(ctx)
            kept, dropped, split = [], 0, False
            for claim in claims:
                text = claim_text(claim)
                grounded = self._ground(ev, claim, text) if text else None
                if grounded is not None:
                    kept.append(grounded)
                    continue
                halves = self._split(ev, text) if text else None
                if halves:
                    kept.extend(halves)
                    split = True  # hai nguồn mâu thuẫn -> thận trọng
                    continue
                dropped += 1  # không có trên dòng nào của thứ đã đọc -> bịa
            kept = self._cap(kept)
            ctx.state["critic_dropped"] = dropped + len(claims) - len(kept)
            report["claims"] = kept
            if split:
                report["abstain"] = True
            if not kept:
                report["abstain"] = True
                report["answer"] = "Không đủ căn cứ trong các tài liệu đã đọc để trả lời câu hỏi này."
            report["citations"] = sorted({d for d in map(claim_doc_id, kept) if d})
        self._settle_verdict(ctx, report)
        return report
