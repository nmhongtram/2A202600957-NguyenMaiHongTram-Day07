# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Thành viên B]
**Nhóm:** [Tên nhóm]
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

Hai text chunk có high cosine similarity khi các vector embedding của chúng trỏ theo cùng hướng trong không gian nhiều chiều. Điều đó có nghĩa nội dung của hai đoạn mang ngữ nghĩa gần nhau — dù từ ngữ bề mặt có thể khác nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Người lao động được hưởng ít nhất 12 ngày nghỉ hằng năm có lương."
- Sentence B: "Mỗi năm nhân viên có quyền nghỉ phép 12 ngày và được trả lương đầy đủ."
- Tại sao tương đồng: Hai câu cùng diễn đạt một quy định về số ngày nghỉ phép năm, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: "Nhà nước khuyến khích tuần làm việc 40 giờ đối với người lao động."
- Sentence B: "Tổ chức của người lao động tại doanh nghiệp phải đăng ký với cơ quan nhà nước."
- Tại sao khác: Một câu nói về thời gian làm việc, câu kia về thủ tục đăng ký tổ chức đại diện — hoàn toàn khác chủ đề.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Cosine similarity chỉ quan tâm đến hướng của vector, không bị ảnh hưởng bởi độ dài văn bản. Một câu ngắn và một đoạn dài cùng nội dung sẽ có cosine cao dù magnitude rất khác nhau — điều Euclidean không làm được vì nó bị chi phối bởi độ lớn vector.

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

Overlap nhiều hơn làm tăng số chunks vì bước nhảy nhỏ hơn. Overlap giúp đảm bảo rằng thông tin nằm ở ranh giới hai chunks liên tiếp không bị mất — đặc biệt quan trọng với văn bản pháp lý khi một khoản có thể kéo dài qua ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Pháp luật lao động Việt Nam — Bộ luật Lao động 45/2019/QH14

**Tại sao nhóm chọn domain này?**

Bộ luật Lao động có cấu trúc pháp lý phân cấp rõ ràng (Chương → Điều → Khoản), mỗi điều khoản chứa thông tin cụ thể và có thể verify rõ ràng. Benchmark queries dễ xây dựng và gold answers chính xác, không mơ hồ. Domain thực tế và có nhu cầu ứng dụng RAG trong tư vấn pháp luật lao động.

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
| `topic` | str | `thoi-gio-lam-viec` | Filter câu hỏi về giờ làm việc chỉ trong chương liên quan |
| `chapters` | str | `VII-IX` | Hỗ trợ retrieval theo phạm vi chương của luật |
| `source` | str | `BLLĐ_45-2019` | Phân biệt nguồn khi mở rộng thêm văn bản pháp luật khác |
| `doc_position` | str | `early` / `mid` / `late` | Biết vị trí tương đối trong văn bản — có thể dùng để ưu tiên các Điều đầu tiên (quy định chung) |
| `dieu_range` | str | `"105,106"` | Số điều khoản trong chunk — filter chính xác đến từng Điều |
| `chunk_len` | int | `395` | Độ dài chunk — debug và loại bỏ chunk quá ngắn |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=500` trên Chương VII-IX:

| Strategy | Chunks | Avg | Min | Max | Kết luận |
|----------|--------|-----|-----|-----|----------|
| FixedSizeChunker(500,0) | 43 | 493 | 195 | 500 | Nhất quán nhất |
| SentenceChunker(3) | 67 | 315 | 16 | 1426 | Giữ câu, nhưng bất nhất |
| RecursiveChunker(default) | 467 | 44 | 1 | 500 | Kém — text không có `\n\n` nên chunk siêu nhỏ |

### Strategy Của Tôi (v2 — redesigned)

**Loại:** FixedSizeChunker với `chunk_size=400, overlap=100` (ratio 25%)

**Thay đổi so với v1 (600, overlap=80, ratio≈13%):**
- Giảm chunk_size 600 → 400: chunk focused hơn, phù hợp với Điều luật ngắn (50-400 chars)
- Tăng overlap ratio 13% → 25%: bảo toàn context tốt hơn ở ranh giới

**Metadata mới:** `doc_position` (early/mid/late), `dieu_range`, `chunk_len`

```python
def chunk_strategy_b(text, topic, chapters):
    chunks = FixedSizeChunker(chunk_size=400, overlap=100).chunk(text)
    docs, offset = [], 0
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            id=f'B_{topic}_c{i}', content=chunk,
            metadata={
                'strategy': 'B_fixed', 'topic': topic, 'chapters': chapters,
                'source': 'BLLĐ_45-2019',
                'doc_position': doc_position(offset, len(text)),
                'dieu_range': extract_dieu_range(chunk),
                'chunk_len': len(chunk),
            }
        ))
        offset += 300  # step = 400 - 100
    return docs
```

**Tại sao FixedSizeChunker vẫn tốt cho pháp lý?** Chunk 400 chars ≈ 1-2 khoản luật — đủ context để embed mà không pha loãng. Overlap 25% bảo đảm continuity. `doc_position` hữu ích khi muốn ưu tiên các Điều đầu văn bản (quy định chung) trong retrieval.

### So Sánh 3 Strategy — Số Liệu Thực với LocalEmbedder

| Strategy | Tham số | Chunks | Avg | Min | Max | Precision@3 |
|----------|---------|--------|-----|-----|-----|-------------|
| A SentenceChunker+merge | max=3, merge<150 | 483 | 390 | 37 | 2081 | 3/5 |
| **B (tôi) FixedSizeChunker** | **400/100** | **632** | **398** | **122** | **400** | **4/5** |
| C RecursiveChunker | ['\n\n','.',' '],500 | 10,204 | 17 | 1 | 496 | 2/5 |

**Score filtered (LocalEmbedder):**

| Query | A | **B** | C |
|-------|---|---|---|
| Q1 Thời giờ làm việc | 0.6632 | **0.6946** | 0.7490 |
| Q2 Giờ làm việc ban đêm | 0.7980 | 0.7588 | 0.8661 |
| Q3 Trách nhiệm đào tạo | 0.8221 | 0.8306 | 0.9198 |
| Q4 Nội dung thanh tra | 0.7423 | **0.7520** | 0.8974 |
| Q5 NLLĐ nước ngoài | **0.8816** | 0.8557 | 0.8636 |
| **Avg** | 0.7814 | 0.7783 | **0.8592** |

**Strategy B dẫn đầu Precision@3 (4/5) dù score thấp hơn C.** Đây là minh chứng cho "micro-chunk score inflation": Strategy C avg 17 chars — chunk tiêu đề "Nội dung thanh tra lao động\n1" score 0.8974 nhưng thiếu nội dung thực tế của Điều 214.

**Strategy tốt nhất:** B cho retrieval đáng tin cậy nhất. A tốt hơn B khi chunk tình cờ chứa trọn vẹn một Điều. C cần fix: loại bỏ `' '` space separator, chỉ dùng `['\n', '. ']` hoặc custom article-based chunking.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

Dùng `re.split(r'(?<=[.!?]) +|(?<=\.)\n', text)` để tách text tại điểm sau dấu kết thúc câu. Các fragment rỗng bị lọc, sau đó nhóm N câu liên tiếp vào một chunk bằng join(' '). Edge case quan trọng: nếu không tìm thấy sentence boundary nào, trả về `[text]` thay vì list rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:

Base case: text ngắn hơn `chunk_size` → trả về ngay. Recursive case: thử từng separator theo thứ tự ưu tiên, split text, nếu phần nào vẫn quá dài thì đệ quy với separator tiếp theo. Separator `""` là last resort: cắt cứng theo ký tự. Key insight: thứ tự separator quyết định "hạt nhân" của chunk — separator đầu tiên phải phù hợp với cấu trúc văn bản.

### EmbeddingStore

**`add_documents` + `search`** — approach:

`_make_record()` gọi `embedding_fn(doc.content)` để tạo vector, lưu cùng `id`, `content`, `metadata` (có thêm `doc_id`). `search()` gọi `_search_records()` tính dot product giữa query embedding và tất cả stored embeddings, sort descending, return top_k. Dùng dot product thay vì cosine vì `_mock_embed` trả về unit vectors — dot product = cosine khi vectors đã normalized.

**`search_with_filter` + `delete_document`** — approach:

`search_with_filter()` tạo danh sách phụ với list comprehension kiểm tra tất cả key-value trong `metadata_filter`. `delete_document()` dùng list comprehension loại bỏ records có `metadata['doc_id'] == doc_id`, compare size để biết có xóa được không.

### KnowledgeBaseAgent

**`answer`** — approach:

Retrieve top-k chunks, format thành numbered context `[1] ... [2] ...`, build prompt `Context:\n...\n\nQuestion: ...\n\nAnswer:` rồi pass cho `llm_fn`. Format này theo RAG pattern chuẩn — LLM được hướng dẫn rõ ràng phải dựa vào context được cung cấp.

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
| 2 | "Giờ làm việc ban đêm tính từ 22 giờ đến 06 giờ sáng." | "Thanh tra lao động có thẩm quyền xử lý vi phạm." | LOW | 0.5250 | Gần đúng — medium, không thấp |
| 3 | "Người sử dụng lao động xây dựng kế hoạch đào tạo hằng năm." | "Doanh nghiệp phải lên kế hoạch bồi dưỡng kỹ năng mỗi năm." | HIGH | 0.5485 | Một phần — medium-high |
| 4 | "Người lao động có quyền nghỉ hằng năm ít nhất 12 ngày." | "Làm thêm giờ không quá 40 giờ trong 01 tháng." | LOW | 0.4875 | ✓ — thấp hơn pair 3 |
| 5 | "Bộ luật Lao động quy định tiêu chuẩn lao động." | "Luật lao động đặt ra các quy định về quan hệ lao động." | HIGH | **0.8393** | ✓ — rất cao |

**Kết quả nào bất ngờ nhất?**

Pair 2 bất ngờ nhất: dự đoán LOW nhưng score 0.5250 — vì cả hai câu đều trong domain pháp lý lao động, embedding nắm bắt được ngữ cảnh domain. Pair 5 (0.8393) và Pair 1 (0.7175) đúng như kỳ vọng. Điều này cho thấy LocalEmbedder hiểu được cả semantic similarity lẫn domain-level similarity — quan trọng cho RAG trong pháp lý vì nhiều điều khoản dùng từ ngữ khác nhau nhưng cùng chủ đề.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer | Điều khoản nguồn |
|---|-------|-------------|-----------------|
| 1 | Thời giờ làm việc bình thường tối đa là bao nhiêu giờ mỗi ngày và mỗi tuần? | Không quá 08 giờ/ngày và 48 giờ/tuần; khuyến khích 40 giờ/tuần | Điều 105 – Chương VII |
| 2 | Giờ làm việc ban đêm được tính từ mấy giờ đến mấy giờ? | Từ 22 giờ đến 06 giờ sáng ngày hôm sau | Điều 106 – Chương VII |
| 3 | Người sử dụng lao động có trách nhiệm gì về đào tạo bồi dưỡng kỹ năng nghề? | Xây dựng kế hoạch hằng năm + dành kinh phí đào tạo; thông báo kết quả hằng năm | Điều 60 – Chương IV |
| 4 | Thanh tra lao động có những nội dung thanh tra gì? | (1) Chấp hành pháp luật; (2) Điều tra tai nạn; (3) Hướng dẫn tiêu chuẩn; (4) Giải quyết khiếu nại; (5) Xử lý vi phạm | Điều 214 – Chương XVI |
| 5 | Người lao động nước ngoài có thuộc đối tượng áp dụng Bộ luật Lao động không? | Có, Điều 2 khoản 3 | Điều 2 – Chương I |

### Kết Quả Của Tôi — Strategy B v2: FixedSizeChunker(400, overlap=100)

**Embedder: all-MiniLM-L6-v2 | Corpus: 632 chunks | Avg: 398 chars | Range: 122–400 chars**

| # | Query | Top-1 Filtered | Score | Relevant? |
|---|-------|----------------|-------|-----------|
| 1 | Thời giờ làm việc tối đa? | "báo với người sử dụng lao động khi ông nội, bà nội... chết; cha hoặc mẹ..." | 0.6946 | ✗ top-1 — nhưng top-3 chứa chunk về Điều 105 |
| 2 | Giờ làm việc ban đêm? | "...nghỉ giữa giờ ít nhất 30 phút liên tục, làm việc ban đêm thì được nghỉ giữa giờ ít nhất 45 phút..." | **0.7588** | ✓ — chunk liên quan đến giờ làm việc ban đêm |
| 3 | Trách nhiệm đào tạo kỹ năng? | "...tham gia hội đồng kỹ năng nghề; dự báo nhu cầu và xây dựng tiêu chuẩn kỹ năng nghề..." | **0.8306** | ✓ — chunk về kỹ năng nghề, Điều 60 |
| 4 | Nội dung thanh tra lao động? | "Chương XVI THANH TRA LAO ĐỘNG... Điều 214. Nội dung thanh tra lao động / 1. Th..." | **0.7520** | ✓ — top filtered chứa toàn bộ Điều 214 |
| 5 | NLLĐ nước ngoài có bị áp dụng? | "...làm việc cho cá nhân là công dân nước ngoài tại Việt Nam phải tuân theo pháp luật..." | 0.8557 | ✗ — topic đúng nhưng không phải Điều 2 khoản 3 |

**Precision@3 (filtered): 4/5** | **Avg score: 0.7783**

**Q1 chi tiết:** Top-1 filtered không relevant (chunk về nghỉ tang), nhưng top-3 chứa "thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày" → Precision@3 = ✓.

**Q5 failure:** Filter `topic=quy-dinh-chung` đúng nhưng top-1 là chunk về người nước ngoài làm việc cho ngoại giao, không phải Điều 2 đối tượng áp dụng. Nguyên nhân: "nước ngoài" xuất hiện nhiều chỗ trong chương I, overlap 100 chars không đủ để phân biệt.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

Từ Strategy C: 10,204 chunk avg 17 chars vì corpus không có `\n\n` — bài học về tầm quan trọng của việc phân tích cấu trúc văn bản trước khi chọn separator. Từ Strategy A: SentenceChunker + merge cho chunk có nghĩa pháp lý hơn (câu trọn vẹn) nhưng min=37 chars vẫn còn nhỏ. Kết hợp hai insight: FixedSizeChunker với overlap 25% và min chunk size ~150 là điểm cân bằng tốt nhất.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

[Điền sau khi demo]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

Khi extract docx → txt sẽ thêm `\n\n` giữa các Điều (dùng regex thay thế `\nĐiều` → `\n\nĐiều`) để cả 3 strategy đều hưởng lợi từ paragraph-level structure. Ngoài ra sẽ dùng `dieu_range` để filter chính xác hơn — thay vì chỉ `topic=quy-dinh-chung`, dùng `{'topic':'quy-dinh-chung', 'dieu_range':'2'}` để fix Q5.

### Failure Case Analysis

**Failure:** Q5 (NLLĐ nước ngoài) với Strategy B filtered — top-1 là chunk về "làm việc cho cá nhân là công dân nước ngoài" chứ không phải Điều 2 khoản 3.

**Nguyên nhân:** FixedSizeChunker tạo nhiều chunk 400-char chứa từ "nước ngoài" trong chương I. Overlap 100 không đủ để phân biệt chunk "đối tượng áp dụng" vs "giấy phép lao động". Embedding thấy "nước ngoài + lao động + Việt Nam" trong nhiều chunk với score tương đồng.

**Đề xuất:** Dùng `search_with_filter({'topic':'quy-dinh-chung', 'dieu_range':'2'})` để target đúng Điều 2. Điều này chứng minh `dieu_range` metadata quan trọng hơn `doc_position` cho domain pháp lý.

**Đề xuất:** Dùng overlap lớn hơn (150 chars) và kết hợp với metadata `dieu_so` để filtered search có thể truy cập trực tiếp Điều 106.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 7 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | [chờ demo] |
| **Tổng** | | **~78 / 100** |
