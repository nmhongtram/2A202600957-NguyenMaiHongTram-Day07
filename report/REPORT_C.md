# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Thành viên C]
**Nhóm:** [Tên nhóm]
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

Hai text chunk có high cosine similarity khi góc giữa hai vector embedding của chúng nhỏ, tức là chúng "trỏ về cùng hướng" trong không gian embedding. Về mặt thực tế, điều này có nghĩa hai đoạn văn có ngữ nghĩa hoặc chủ đề gần giống nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Người sử dụng lao động phải trả lương đúng hạn cho người lao động."
- Sentence B: "Tiền lương phải được thanh toán đầy đủ và đúng kỳ hạn cho nhân viên."
- Tại sao tương đồng: Hai câu đều diễn đạt nghĩa vụ trả lương đúng hạn, khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: "Hội đồng trọng tài lao động giải quyết tranh chấp lao động tập thể."
- Sentence B: "Người lao động nữ được nghỉ thai sản 06 tháng có lương."
- Tại sao khác: Một câu về cơ chế giải quyết tranh chấp, câu kia về chế độ thai sản — hai mảng pháp lý hoàn toàn khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Văn bản dài thường có embedding vector có magnitude lớn hơn văn bản ngắn. Euclidean distance bị ảnh hưởng bởi magnitude này, nên câu ngắn và câu dài cùng nghĩa lại có khoảng cách Euclidean lớn. Cosine chỉ đo góc nên không bị vấn đề này.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

```
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
           = ceil((10000 - 50) / (500 - 50))
           = ceil(9950 / 450)
           = ceil(22.11)
           = 23 chunks
```

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

```
num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25 chunks
```

Tăng overlap → nhiều chunks hơn vì bước nhảy nhỏ hơn. Overlap nhiều có giá trị khi văn bản có "transition sentences" quan trọng nằm ở ranh giới — ví dụ kết luận của một khoản luật có thể là đầu vào cho khoản tiếp theo.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Pháp luật lao động Việt Nam — Bộ luật Lao động 45/2019/QH14

**Tại sao nhóm chọn domain này?**

Domain pháp lý có cấu trúc văn bản rõ ràng và nhất quán, phù hợp để đánh giá các chunking strategy khác nhau. Queries benchmark dễ xây dựng và gold answers chính xác. Ứng dụng RAG trong tư vấn pháp luật lao động có nhu cầu thực tế cao, đặc biệt với doanh nghiệp và người lao động.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | chuong1-3_quy-dinh-chung.txt | BLLĐ 45/2019 Chương I-III | 51,501 | topic=quy-dinh-chung, chapters=I-III |
| 2 | chuong4-6_giao-duc-nghe-nghiep.txt | BLLĐ 45/2019 Chương IV-VI | 37,622 | topic=giao-duc-nghe-nghiep, chapters=IV-VI |
| 3 | chuong7-9_thoi-gio-lam-viec.txt | BLLĐ 45/2019 Chương VII-IX | 21,195 | topic=thoi-gio-lam-viec, chapters=VII-IX |
| 4 | chuong10-12_lao-dong-nu.txt | BLLĐ 45/2019 Chương X-XII | 26,185 | topic=lao-dong-nu, chapters=X-XII |
| 5 | chuong13-15_to-chuc-dai-dien.txt | BLLĐ 45/2019 Chương XIII-XV | 41,602 | topic=to-chuc-dai-dien, chapters=XIII-XV |
| 6 | chuong16-17_thanh-tra.txt | BLLĐ 45/2019 Chương XVI-XVII | 10,963 | topic=thanh-tra, chapters=XVI-XVII |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `topic` | str | `thanh-tra` | Filter chính xác đến Chương/chủ đề phù hợp với câu hỏi |
| `chapters` | str | `XVI-XVII` | Tra cứu theo phạm vi chương của văn bản luật |
| `source` | str | `BLLĐ_45-2019` | Phân biệt nguồn khi kết hợp nhiều văn bản |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên Chương VII-IX (21,195 chars):

| Strategy | Chunks | Avg | Min | Max | Kết luận |
|----------|--------|-----|-----|-----|----------|
| FixedSizeChunker(500,0) | 43 | 493 | 195 | 500 | Ổn định |
| SentenceChunker(3) | 67 | 315 | 16 | 1426 | Giữ câu, bất nhất |
| RecursiveChunker(default) | 467 | 44 | 1 | 500 | Kém |

### Strategy Của Tôi (v2 — redesigned)

**Loại:** RecursiveChunker với separators `['\n\n', '. ', ' ']`, chunk_size=500

**Thay đổi so với v1 (['\n\n', '\n', '. '], 600):**
- Bỏ `'\n'` đơn để tránh tạo chunk quá nhỏ (v1 tạo 1341 chunks avg 140)
- Bỏ `' '` space để làm last resort... **nhưng kết quả v2 vẫn tạo 10,204 chunks** (phân tích bên dưới)

**Root cause được phát hiện sau khi chạy:**

```python
# Kiểm tra cấu trúc file txt:
text.count('\n\n')  # = 0  !!!
text.count('\n')    # = 171
text.count('. ')    # = 117
text.count(' ')     # = 4527
```

File txt được extract từ docx **không có `\n\n`** — toàn bộ dùng `\n` đơn. Khi RecursiveChunker thử `'\n\n'` → không tìm thấy → nhảy xuống `'. '` → tạo sentence fragments → nếu fragment > 500 chars → thử `' '` (space) → tách từng từ. Kết quả: 10,204 chunk avg 17 chars!

**Metadata mới:** `starts_with_article` (bool), `dieu_range`, `chunk_len`

```python
def chunk_strategy_c(text, topic, chapters):
    chunks = RecursiveChunker(
        separators=['\n\n', '. ', ' '],
        chunk_size=500,
    ).chunk(text)
    return [Document(
        id=f'C_{topic}_c{i}', content=chunk,
        metadata={
            'strategy': 'C_paragraph', 'topic': topic, 'chapters': chapters,
            'source': 'BLLĐ_45-2019',
            'dieu_range': extract_dieu_range(chunk),
            'starts_with_article': bool(re.match(r'^\s*Điều\s+\d+', chunk)),
            'chunk_len': len(chunk),
        }
    ) for i, chunk in enumerate(chunks)]
```

**Cách fix đúng:** Khi extract docx → txt, thêm `\n\n` trước mỗi "Điều":
```python
text = re.sub(r'(\nĐiều\s+\d+)', r'\n\n\1', text)
# Hoặc dùng separators=['\n', '. '] — bỏ ' ' space để tránh word-level splitting
```

### So Sánh 3 Strategy — Số Liệu Thực với LocalEmbedder

| Strategy | Tham số | Chunks | Avg | Min | Max | Precision@3 |
|----------|---------|--------|-----|-----|-----|-------------|
| A SentenceChunker+merge | max=3, merge<150 | 483 | 390 | 37 | 2081 | 3/5 |
| B FixedSizeChunker | 400/100 | 632 | 398 | 122 | 400 | 4/5 |
| **C (tôi) RecursiveChunker** | **['\n\n','.',' '],500** | **10,204** | **17** | **1** | **496** | **2/5** |

**Score filtered (LocalEmbedder):**

| Query | A | B | **C** |
|-------|---|---|---|
| Q1 Thời giờ làm việc | 0.6632 | 0.6946 | **0.7490** |
| Q2 Giờ làm việc ban đêm | 0.7980 | 0.7588 | **0.8661** |
| Q3 Trách nhiệm đào tạo | 0.8221 | 0.8306 | **0.9198** |
| Q4 Nội dung thanh tra | 0.7423 | 0.7520 | **0.8974** |
| Q5 NLLĐ nước ngoài | **0.8816** | 0.8557 | 0.8636 |
| **Avg** | 0.7814 | 0.7783 | **0.8592** |

**Paradox Strategy C:** Scores cao nhất (avg 0.8592) nhưng precision thấp nhất (2/5). Lý do: chunk tiêu đề 10-30 chars như "Nội dung thanh tra lao động" khớp keyword query rất tốt → cosine cao, nhưng chunk không chứa nội dung thực sự. Đây là **"micro-chunk score inflation"** — cần phân biệt giữa *similar* và *informative*.

**Kết luận:** Strategy B đáng tin cậy nhất. Strategy C cần fix corpus extraction trước khi có thể đánh giá thực sự.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

Split bằng regex `(?<=[.!?]) +|(?<=\.)\n` để tách tại điểm sau dấu kết thúc câu. Sau đó nhóm `max_sentences_per_chunk` câu thành một chunk, join bằng space. Quan trọng: filter các fragment rỗng (`if s.strip()`) để tránh chunk trống từ các đoạn có nhiều dấu xuống dòng.

**`RecursiveChunker.chunk` / `_split`** — approach:

Thiết kế divide-and-conquer: nếu text fit trong `chunk_size` → done; nếu không, tìm separator tốt nhất để chia, rồi đệ quy xử lý phần nào còn quá lớn. Separator `""` là safety net cuối cùng: cắt cứng theo ký tự. Điểm mấu chốt: khi separator không tìm thấy trong text, chuyển sang separator tiếp theo thay vì bỏ cuộc.

### EmbeddingStore

**`add_documents` + `search`** — approach:

Mỗi Document được nhúng thành vector qua `embedding_fn`, lưu kèm `id`, `content`, `metadata` (với `doc_id` là bản sao của `id` để hỗ trợ delete). `search()` embed query, tính dot product với tất cả stored vectors (hiệu quả vì vectors đã normalized thành unit vectors), sort và slice top_k.

**`search_with_filter` + `delete_document`** — approach:

Filter-first strategy trong `search_with_filter()`: list comprehension lọc records match tất cả key-value trong `metadata_filter`, rồi mới tính similarity — giảm số phép tính và tăng precision. `delete_document()` immutable approach: tạo list mới loại bỏ records có `doc_id` match, return True/False dựa trên thay đổi kích thước.

### KnowledgeBaseAgent

**`answer`** — approach:

RAG pattern: (1) retrieve top-k chunks từ store, (2) format context thành numbered list `[1]...[2]...`, (3) build prompt `Context:\n...\n\nQuestion: {q}\n\nAnswer:`, (4) call LLM. Numbered context giúp LLM có thể reference `[1]` hay `[2]` trong câu trả lời để tăng traceability.

### Test Results

```
============================= test session starts =============================
collected 42 items

[tất cả 42 tests PASSED]

============================== 42 passed in 0.05s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Embedder: `all-MiniLM-L6-v2` (LocalEmbedder)*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Thời giờ làm việc bình thường không quá 08 giờ/ngày." | "Người lao động làm không quá 8 tiếng mỗi ngày." | HIGH | **0.7175** | ✓ |
| 2 | "Giờ làm việc ban đêm tính từ 22 giờ đến 06 giờ sáng." | "Thanh tra lao động có thẩm quyền xử lý vi phạm." | LOW | 0.5250 | Gần đúng — medium |
| 3 | "Người sử dụng lao động xây dựng kế hoạch đào tạo hằng năm." | "Doanh nghiệp phải lên kế hoạch bồi dưỡng kỹ năng mỗi năm." | HIGH | 0.5485 | Một phần |
| 4 | "Người lao động có quyền nghỉ hằng năm ít nhất 12 ngày." | "Làm thêm giờ không quá 40 giờ trong 01 tháng." | LOW | 0.4875 | ✓ |
| 5 | "Bộ luật Lao động quy định tiêu chuẩn lao động." | "Luật lao động đặt ra các quy định về quan hệ lao động." | HIGH | **0.8393** | ✓ |

**Kết quả nào bất ngờ nhất?**

Pair 5 (0.8393) cao hơn Pair 1 (0.7175) dù Pair 1 trực tiếp hơn. Lý do: Pair 5 gần đồng nghĩa hoàn toàn về mặt từ ngữ; Pair 1 đổi đơn vị "08 giờ" → "8 tiếng" — model chưa học được "giờ" = "tiếng" trong tiếng Việt. Pair 2 (0.5250) medium thay vì LOW — embedding nhận ra cả hai câu đều thuộc domain pháp lý lao động.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer | Điều khoản nguồn |
|---|-------|-------------|-----------------|
| 1 | Thời giờ làm việc bình thường tối đa là bao nhiêu giờ mỗi ngày và mỗi tuần? | Không quá 08 giờ/ngày và 48 giờ/tuần; khuyến khích 40 giờ/tuần | Điều 105 – Chương VII |
| 2 | Giờ làm việc ban đêm được tính từ mấy giờ đến mấy giờ? | Từ 22 giờ đến 06 giờ sáng ngày hôm sau | Điều 106 – Chương VII |
| 3 | Người sử dụng lao động có trách nhiệm gì về đào tạo bồi dưỡng kỹ năng nghề? | Xây dựng kế hoạch hằng năm + dành kinh phí đào tạo; thông báo kết quả | Điều 60 – Chương IV |
| 4 | Thanh tra lao động có những nội dung thanh tra gì? | (1) Chấp hành pháp luật; (2) Điều tra tai nạn; (3) Hướng dẫn tiêu chuẩn; (4) Giải quyết khiếu nại; (5) Xử lý vi phạm | Điều 214 – Chương XVI |
| 5 | Người lao động nước ngoài có thuộc đối tượng áp dụng Bộ luật Lao động không? | Có, Điều 2 khoản 3 | Điều 2 – Chương I |

### Kết Quả Của Tôi — Strategy C v2: RecursiveChunker(['\n\n','.',' '], chunk_size=500)

**Embedder: all-MiniLM-L6-v2 | Corpus: 10,204 chunks | Avg: 17 chars | Range: 1–496 chars**

| # | Query | Top-1 Filtered | Score | Relevant? | Ghi chú |
|---|-------|----------------|-------|-----------|---------|
| 1 | Thời giờ làm việc tối đa? | "THỜI GIỜ LÀM VIỆC, THỜI GIỜ NGHỈ NGƠI ĐỐI VỚI NGƯỜI LÀM CÔNG VIỆC CÓ TÍNH CHẤT ĐẶC BIỆT / Điều 116" | **0.7490** | ✗ | Tiêu đề chương đặc biệt, không phải Điều 105 |
| 2 | Giờ làm việc ban đêm? | "Giờ làm việc ban đêm / Giờ làm việc ban đêm được tính từ 22 giờ đến 06 giờ sáng ngày hôm sau. / Điều 107" | **0.8661** | ✓ | Exact match — chunk nhỏ nhưng chứa đúng nội dung Điều 106 |
| 3 | Trách nhiệm đào tạo kỹ năng? | "Trách nhiệm của người sử dụng lao động về đào tạo, bồi dưỡng, nâng cao trình độ, kỹ năng nghề / 1" | **0.9198** | ✓ | Tiêu đề Điều 60 — score rất cao nhưng thiếu nội dung khoản 1,2 |
| 4 | Nội dung thanh tra lao động? | "Nội dung thanh tra lao động / 1" | **0.8974** | ✗ | Chỉ tiêu đề + "1", thiếu 5 nội dung thanh tra thực tế |
| 5 | NLLĐ nước ngoài có bị áp dụng? | "Bảo đảm trả lương cho người lao động thuê lại không thấp hơn tiền lương..." | 0.8636 | ✗ | Chunk về lao động thuê lại, không phải Điều 2 |

**Precision@3 (filtered): 2/5** | **Avg score: 0.8592 — CAO NHẤT nhưng misleading**

**Phân tích "Micro-chunk Score Inflation":**

Q3 (score 0.9198): chunk là "Trách nhiệm của người sử dụng lao động về đào tạo, bồi dưỡng, nâng cao trình độ, kỹ năng nghề\n1" (~90 chars). Cosine rất cao vì chunk title khớp hoàn hảo với query, nhưng nội dung thực ("Xây dựng kế hoạch hằng năm và dành kinh phí...") nằm ở chunk tiếp theo — RAG agent dùng chunk này sẽ **không có đủ thông tin để trả lời đúng**.

Q4 (score 0.8974): tương tự — "Nội dung thanh tra lao động\n1" match keyword nhưng không có 5 nội dung cụ thể của Điều 214.

**Kết luận về Strategy C:** Score cao ≠ Chunk informative. Cần fix corpus (`\n\n` trước Điều) và bỏ `' '` space separator.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

Từ Strategy B: overlap 25% (100/400) quan trọng hơn nhiều so với 13% (80/600) — với chunk 400 chars, mất 100 chars context ở ranh giới có thể làm mất trọn câu quan trọng. Từ Strategy A: SentenceChunker + merge approach hay hơn chỉ SentenceChunker thuần — merge giải quyết được fragment nhỏ mà không cần min_chunk_size parameter. **Quan trọng nhất:** Cả 3 strategy đều hưởng lợi nếu corpus có `\n\n` — đây là bài học về data preparation quan trọng hơn algorithm tuning.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

[Điền sau khi demo]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

**Bước 1 — Fix corpus extraction:** Thêm `\n\n` trước mỗi "Điều" khi extract docx → txt:
```python
text = re.sub(r'\n(Điều\s+\d+)', r'\n\n\1', text)
```
Điều này biến 0 double-newlines thành ~250 double-newlines, cho RecursiveChunker hoạt động đúng thiết kế.

**Bước 2 — Fix separators:** Dùng `['\n\n', '\n', '. ']` (bỏ `' '` space), sau khi fix corpus sẽ có chunk avg ~350 chars thay vì 17 chars.

### Failure Case Analysis

**Failure rõ nhất (với LocalEmbedder):** Q3 và Q4 — score rất cao (0.9198, 0.8974) nhưng chunk chỉ là tiêu đề Điều, không có nội dung.

**Nguyên nhân 3 tầng:**
1. **Corpus không có `\n\n`** → RecursiveChunker không split được theo paragraph → fallback xuống `. ` rồi `' '` → 10,204 chunk avg 17 chars
2. **Chunk = tiêu đề** (e.g., "Trách nhiệm của người sử dụng lao động về đào tạo...") khớp hoàn hảo keyword → score cao
3. **Content trong chunk tiếp theo** — RAG agent chỉ dùng top-1, bỏ qua khoản 1,2 chứa câu trả lời thực sự

**Đề xuất (đã verify bằng diagnosis):**
```python
# Bước 1: Re-extract corpus với double newline
text = re.sub(r'\n(Điều\s+\d+)', r'\n\n\1', text)
# Bước 2: RecursiveChunker sẽ split đúng theo paragraph
chunker = RecursiveChunker(separators=['\n\n', '\n', '. '], chunk_size=500)
# Kỳ vọng: ~300-400 chunks avg ~300-400 chars, precision@3 ≥ 4/5
```

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 7 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | [chờ demo] |
| **Tổng** | | **~77 / 100** |
