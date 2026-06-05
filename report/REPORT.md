# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Mai Hồng Trâm
**Nhóm:** [Tên nhóm]
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

Hai text chunk có high cosine similarity khi các vector embedding của chúng trỏ theo cùng hướng trong không gian nhiều chiều, tức là chúng chia sẻ ngữ nghĩa tương tự nhau — dù từ ngữ có khác nhau, ý nghĩa vẫn gần.

**Ví dụ HIGH similarity:**
- Sentence A: "Người lao động không được làm quá 8 giờ mỗi ngày."
- Sentence B: "Thời gian làm việc tối đa là 8 tiếng trong một ngày."
- Tại sao tương đồng: Cả hai diễn đạt cùng một quy định pháp lý về giới hạn giờ làm việc, chỉ khác cách dùng từ.

**Ví dụ LOW similarity:**
- Sentence A: "Giờ làm việc ban đêm tính từ 22 giờ đến 06 giờ sáng."
- Sentence B: "Người sử dụng lao động phải đóng bảo hiểm xã hội cho nhân viên."
- Tại sao khác: Hai câu thuộc hoàn toàn hai chủ đề khác nhau (thời giờ làm việc vs bảo hiểm xã hội).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Cosine similarity đo góc giữa hai vector nên không bị ảnh hưởng bởi độ dài văn bản — một đoạn ngắn và một đoạn dài cùng nội dung vẫn cho similarity cao. Euclidean distance lại nhạy cảm với độ lớn vector, làm cho văn bản dài luôn "xa" hơn văn bản ngắn bất kể nội dung.

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
num_chunks = ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24.75) = 25 chunks
```

Tăng overlap làm tăng số chunks vì mỗi bước tiến nhỏ hơn. Overlap nhiều hơn giúp các chunk kề nhau chia sẻ context — quan trọng khi câu quan trọng nằm ở ranh giới giữa hai chunk, tránh mất thông tin khi tách.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Pháp luật lao động Việt Nam — Bộ luật Lao động 45/2019/QH14

**Tại sao nhóm chọn domain này?**

Bộ luật Lao động có cấu trúc phân cấp rõ ràng (Chương → Điều → Khoản → Điểm), phù hợp để thử nghiệm nhiều chiến lược chunking khác nhau. Domain pháp lý thực tế và các câu hỏi benchmark dễ verify vì có câu trả lời chính xác từ điều khoản cụ thể. Đây cũng là tài liệu quan trọng trong thực tế mà AI hỗ trợ tư vấn pháp luật có thể dùng.

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
| `topic` | str | `thoi-gio-lam-viec` | Filter câu hỏi về giờ làm việc chỉ trong chương liên quan, tránh nhiễu từ chương khác |
| `chapters` | str | `VII-IX` | Cho biết phạm vi chương — hỗ trợ retrieval theo cấu trúc luật |
| `source` | str | `BLLĐ_45-2019` | Phân biệt nguồn khi kết hợp nhiều văn bản pháp luật khác nhau |
| `dieu_range` | str | `"105,106"` | Số điều khoản xuất hiện trong chunk — filter chính xác đến từng Điều |
| `chunk_len` | int | `387` | Độ dài chunk — debug, loại bỏ chunk quá ngắn khi cần |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=500` trên Chương VII-IX (21,195 chars):

| Strategy | Chunks | Avg Length | Min | Max | Kết luận |
|----------|--------|-----------|-----|-----|----------|
| FixedSizeChunker | 43 | 493 | 195 | 500 | Ổn định nhất về size |
| SentenceChunker | 67 | 315 | 16 | 1426 | Giữ câu, nhưng có chunk quá dài |
| RecursiveChunker(default) | 467 | 44 | 1 | 500 | Kém — `\n` đơn dày đặc → 93% chunk < 50 chars |

**Phát hiện quan trọng:** Text được extract từ docx **không có `\n\n`** (double newline) — toàn bộ là `\n` đơn. RecursiveChunker(default) dùng `'\n\n'` là separator đầu tiên nhưng không tìm thấy, nhảy thẳng xuống `'\n'` → mỗi dòng thành một chunk riêng → chunk siêu nhỏ.

### Strategy Của Tôi (v2 — redesigned)

**Loại:** SentenceChunker với `max_sentences=3` + post-processing merge chunk < 150 chars

**Thay đổi so với v1 (max_sentences=5):**
- Giảm từ 5 → 3 câu/chunk để tránh chunk quá dài (v1 max=2153 chars)
- Thêm merge step: chunk < 150 chars được nối với chunk kế tiếp trước khi lưu

**Metadata mới:** `dieu_range` (số điều khoản trong chunk), `chunk_len`

**Code snippet:**
```python
def chunk_strategy_a(text, topic, chapters):
    raw_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
    # Merge consecutive chunks shorter than 150 chars
    merged, buffer = [], ''
    for c in raw_chunks:
        if buffer and len(buffer) < 150:
            buffer = buffer + ' ' + c
        else:
            if buffer:
                merged.append(buffer)
            buffer = c
    if buffer:
        merged.append(buffer)
    # Build Documents with enriched metadata
    return [Document(
        id=f'A_{topic}_c{i}',
        content=chunk,
        metadata={
            'strategy': 'A_sentence', 'topic': topic, 'chapters': chapters,
            'source': 'BLLĐ_45-2019',
            'dieu_range': extract_dieu_range(chunk),  # e.g. "105,106"
            'chunk_len': len(chunk),
        }
    ) for i, chunk in enumerate(merged)]
```

**Tại sao strategy này phù hợp với domain pháp lý?**

Mỗi câu trong văn bản luật thường là một khoản quy định độc lập. Nhóm 3 câu (avg ~390 chars) cho phép một chunk chứa đủ ngữ cảnh của một Điều mà không quá dài. Post-merge loại bỏ fragment < 150 chars là tiêu đề/số thứ tự rời, giúp chunk luôn chứa nội dung có nghĩa. `dieu_range` metadata cho phép query "Điều 105 quy định gì?" filter ngay đến chunk liên quan.

### So Sánh 3 Strategy — Số Liệu Thực với LocalEmbedder

**Thống kê chunk toàn corpus (6 phần):**

| Strategy | Tham số | Chunks | Avg | Min | Max | Metadata mới |
|----------|---------|--------|-----|-----|-----|--------------|
| **A** SentenceChunker(3) + merge | max=3, merge<150 | 483 | 390 | 37 | 2081 | dieu_range, chunk_len |
| **B** FixedSizeChunker | chunk_size=400, overlap=100 | 632 | 398 | 122 | 400 | doc_position, dieu_range, chunk_len |
| **C** RecursiveChunker | ['\n\n','.',' '], chunk_size=500 | **10,204** | **17** | **1** | 496 | starts_with_article, dieu_range, chunk_len |

**Quan sát Strategy C:** Vẫn còn 10,204 chunks (avg 17 chars) — separator `' '` (space) là last resort, khi `. ` tạo các sentence fragment ≤ 500 chars thì không cần split tiếp, nhưng vì text có rất ít `\n\n` nên mọi text đều đi qua `\n` và `. ` trước khi đến ` `. Root cause: `. ` splits "1. Người lao động..." thành "1" + "Người lao động..." → "1" < 500 → thành chunk 1 ký tự.

**Kết quả benchmark — LocalEmbedder (semantic scores):**

| Query | A (filtered) | B (filtered) | C (filtered) | Ghi chú |
|-------|-------------|-------------|-------------|---------|
| Q1 Thời giờ làm việc | 0.6632 | **0.6946** | 0.7490 | C cao hơn nhưng chunk là tiêu đề chương |
| Q2 Giờ làm việc ban đêm | **0.7980** | 0.7588 | **0.8661** | A & C lấy được chunk chứa Điều 106 |
| Q3 Trách nhiệm đào tạo | 0.8221 | 0.8306 | **0.9198** | C lấy được tiêu đề Điều 60, thiếu nội dung |
| Q4 Nội dung thanh tra | 0.7423 | 0.7520 | **0.8974** | A & B lấy Điều 214 đầy đủ; C chỉ lấy tiêu đề |
| Q5 NLLĐ nước ngoài | **0.8816** | 0.8557 | 0.8636 | A top-1 exact match khoản 3 Điều 2 |

**Precision@3 (filtered) — keyword-based:**

| Strategy | Q1 | Q2 | Q3 | Q4 | Q5 | **Tổng** |
|----------|----|----|----|----|----|----|
| A | 0 | ✓ | ✓ | ✓ | 0 | **3/5** |
| B | ✓ | ✓ | ✓ | ✓ | 0 | **4/5** |
| C | 0 | ✓ | ✓ | 0 | 0 | **2/5** |

**Phân tích:** Strategy B dẫn đầu về precision (4/5). Strategy C có score cao nhất do "micro-chunk score inflation" — chunk nhỏ chứa đúng keyword nên cosine similarity rất cao, nhưng chunk thiếu nội dung thực (chỉ có tiêu đề, không có khoản giải thích). Strategy A cân bằng tốt giữa score và precision.

**Strategy nào tốt nhất và tại sao:**

Strategy B (FixedSizeChunker 400/100) đáng tin cậy nhất: precision 4/5, chunk size nhất quán (122–400), overlap 25% bảo đảm không mất thông tin ở ranh giới. Strategy A (3/5) tốt cho các query về một Điều cụ thể vì chunk giữ câu trọn vẹn. Strategy C cần sửa: bỏ `' '` space separator, dùng `['\n', '. ']` hoặc custom article-based chunking — đây là bài học hay nhất của nhóm.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:

Dùng `re.split(r'(?<=[.!?]) +|(?<=\.)\n', text)` để tách tại điểm sau dấu kết thúc câu mà không xóa mất delimiter. Các fragment rỗng bị lọc bỏ bằng `if s.strip()`. Sau đó nhóm N câu liên tiếp bằng list slicing và join bằng space.

**`RecursiveChunker.chunk` / `_split`** — approach:

Base case: nếu text ngắn hơn `chunk_size`, trả về `[text]`. Với mỗi separator theo thứ tự ưu tiên: thử split text, nếu separator không có trong text thì chuyển sang separator tiếp theo. Với các phần đã split, nếu phần nào vẫn dài hơn `chunk_size`, đệ quy với danh sách separators còn lại. Separator rỗng `""` là last resort: cắt theo fixed-size characters.

### EmbeddingStore

**`add_documents` + `search`** — approach:

`_make_record()` nhúng content qua `embedding_fn` và lưu cả `id`, `content`, `embedding`, `metadata` (bổ sung `doc_id` vào metadata để hỗ trợ delete sau này). `add_documents()` gọi `_make_record()` cho mỗi doc và append vào `self._store`. `search()` delegate sang `_search_records()` là hàm tính dot product giữa query embedding và tất cả stored embeddings, sort descending, slice top_k.

**`search_with_filter` + `delete_document`** — approach:

`search_with_filter()` filter trước: tạo danh sách phụ `filtered` gồm các record mà tất cả key-value trong `metadata_filter` đều match. Sau đó gọi `_search_records()` trên danh sách phụ này. `delete_document()` dùng list comprehension để loại bỏ tất cả record có `metadata['doc_id'] == doc_id`, so sánh độ dài trước-sau để biết có xóa được không.

### KnowledgeBaseAgent

**`answer`** — approach:

Retrieve top-k chunks từ store theo query. Build prompt gồm phần `Context` liệt kê các chunk đánh số `[1]`, `[2]`... rồi đặt câu hỏi ở cuối theo format `Question: ... \nAnswer:`. Gọi `llm_fn(prompt)` và return kết quả trực tiếp. Pattern này cho LLM biết cần dựa vào context đã cho thay vì tự sinh câu trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-9.0.3
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 0.05s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Embedder: `all-MiniLM-L6-v2` (LocalEmbedder)*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày." | "Người lao động làm không quá 8 tiếng mỗi ngày." | HIGH | **0.7175** | ✓ |
| 2 | "Giờ làm việc ban đêm tính từ 22 giờ đến 06 giờ sáng." | "Thanh tra lao động có thẩm quyền xử lý vi phạm." | LOW | 0.5250 | Gần đúng — medium, không thấp như kỳ vọng |
| 3 | "Người sử dụng lao động xây dựng kế hoạch đào tạo hằng năm." | "Doanh nghiệp phải lên kế hoạch bồi dưỡng kỹ năng mỗi năm." | HIGH | 0.5485 | Một phần — medium, không HIGH như kỳ vọng |
| 4 | "Người lao động có quyền nghỉ hằng năm ít nhất 12 ngày." | "Làm thêm giờ không quá 40 giờ trong 01 tháng." | LOW | 0.4875 | ✓ — thấp hơn pair 3 |
| 5 | "Bộ luật Lao động quy định tiêu chuẩn lao động." | "Luật lao động đặt ra các quy định về quan hệ lao động." | HIGH | **0.8393** | ✓ — rất cao |

**Kết quả nào bất ngờ nhất?**

Pair 2 bất ngờ nhất: tôi dự đoán LOW nhưng score là 0.5250 — vì cả hai câu đều trong domain luật lao động, embedding học được sự tương đồng domain. Pair 3 cũng thú vị: paraphrase trực tiếp nhưng chỉ đạt 0.5485 (không cao như Pair 1 hay 5) — cho thấy model phân biệt được "đào tạo hằng năm" vs "bồi dưỡng kỹ năng" dù hai khái niệm gần nhau. Pair 5 cao nhất (0.8393) vì gần như đồng nghĩa hoàn toàn.

---

## 6. Results — Nhóm (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

Domain: **Bộ luật Lao động 45/2019/QH14**

| # | Query | Gold Answer | Điều khoản nguồn |
|---|-------|-------------|-----------------|
| 1 | Thời giờ làm việc bình thường tối đa là bao nhiêu giờ mỗi ngày và mỗi tuần? | Không quá 08 giờ trong 01 ngày và không quá 48 giờ trong 01 tuần. Nhà nước khuyến khích tuần 40 giờ. | Điều 105 – Chương VII |
| 2 | Giờ làm việc ban đêm được tính từ mấy giờ đến mấy giờ? | Từ 22 giờ đến 06 giờ sáng ngày hôm sau. | Điều 106 – Chương VII |
| 3 | Người sử dụng lao động có trách nhiệm gì về đào tạo, bồi dưỡng kỹ năng nghề cho người lao động? | Phải xây dựng kế hoạch hằng năm và dành kinh phí cho đào tạo, bồi dưỡng, nâng cao trình độ, kỹ năng nghề; hằng năm thông báo kết quả đào tạo cho cơ quan quản lý nhà nước. | Điều 60 – Chương IV |
| 4 | Thanh tra lao động có những nội dung thanh tra gì? | (1) Chấp hành quy định pháp luật về lao động; (2) Điều tra tai nạn lao động và vi phạm an toàn, vệ sinh lao động; (3) Hướng dẫn áp dụng tiêu chuẩn, quy chuẩn kỹ thuật; (4) Giải quyết khiếu nại, tố cáo về lao động; (5) Xử lý vi phạm. | Điều 214 – Chương XVI |
| 5 | Người lao động nước ngoài làm việc tại Việt Nam có thuộc phạm vi áp dụng của Bộ luật Lao động không? | Có. Điều 2 quy định đối tượng áp dụng bao gồm: người lao động nước ngoài làm việc tại Việt Nam (khoản 3). | Điều 2 – Chương I |

> **Ghi chú cho nhóm:** Query 3 và 4 cần metadata filtering (`topic=giao-duc-nghe-nghiep` và `topic=thanh-tra`) để tránh nhiễu từ các chương khác. Đây là trường hợp `search_with_filter()` phát huy tác dụng.

### Kết Quả Của Tôi — Strategy A v2: SentenceChunker(3) + merge<150

**Embedder: all-MiniLM-L6-v2 | Corpus: 483 chunks | Avg: 390 chars | Range: 37–2,081 chars**

| # | Query | Top-1 Filtered | Score | Relevant? |
|---|-------|----------------|-------|-----------|
| 1 | Thời giờ làm việc tối đa? | "Điều 106. Giờ làm việc ban đêm / từ 22 giờ đến 06 giờ sáng. Điều 107..." | 0.6632 | ✗ — retrieves Điều 106 thay vì 105 |
| 2 | Giờ làm việc ban đêm? | "Điều 106. Giờ làm việc ban đêm / từ 22 giờ đến 06 giờ sáng ngày hôm sau. Điều 107..." | **0.7980** | ✓ |
| 3 | Trách nhiệm đào tạo kỹ năng? | "Trách nhiệm của người sử dụng lao động về đào tạo, bồi dưỡng, nâng cao trình độ..." | **0.8221** | ✓ |
| 4 | Nội dung thanh tra lao động? | "Chương XVI THANH TRA LAO ĐỘNG... Điều 214. Nội dung thanh tra lao động / 1. Th..." | **0.7423** | ✓ |
| 5 | NLLĐ nước ngoài có bị áp dụng? | "Người lao động nước ngoài làm việc tại Việt Nam. 4. Cơ quan, tổ chức..." | 0.8816 | ✗ — thiếu "Điều 2" và "đối tượng áp dụng" trong chunk |

**Precision@3 (filtered): 3/5** | **Avg score: 0.7814**

**Failure case Q1 (với LocalEmbedder):** Query "tối đa... mỗi ngày và mỗi tuần" kéo về chunk Điều 106 (ban đêm). Nguyên nhân: từ "ngày", "giờ" xuất hiện cả ở Điều 105 và 106 nên embedding không phân biệt được; cần thêm filter `dieu_range='105'` hoặc tăng top_k và kiểm tra top-3.

**Insight metadata:** `search_with_filter(topic='thanh-tra')` cho Q4 = score 0.7423 và relevant — đây là bằng chứng filter giúp retrieval ngay cả khi score không cao nhất.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**

Strategy C (RecursiveChunker) dạy tôi bài học quan trọng nhất: corpus của chúng ta **không có `\n\n`** — toàn bộ dùng `\n` đơn — nên RecursiveChunker rơi xuống `. ` rồi ` ` space, tạo 10,204 chunk avg 17 chars. Paradox thú vị: C có score cao nhất (avg 0.8592) nhưng precision thấp nhất (2/5) — "micro-chunk score inflation": chunk tiêu đề nhỏ khớp keyword tốt nhưng thiếu nội dung. Strategy B dạy rằng overlap 25% (100/400) hiệu quả hơn 13% (80/600) trong việc bảo toàn context ở ranh giới.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**

[Điền sau khi demo]

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**

Khi extract docx → txt, sẽ thêm `\n\n` giữa các đoạn văn (sau mỗi Điều hoặc Khoản lớn) để RecursiveChunker hoạt động đúng thiết kế. Cũng sẽ thêm metadata `dieu_range` để filter chính xác đến từng Điều — ví dụ `search_with_filter(query, {'topic':'thoi-gio-lam-viec', 'dieu_range':'105'})` giải quyết được failure case Q1.

### Failure Case Analysis (Ex 3.5)

**Query thất bại:** Q1 (Thời giờ làm việc tối đa) với LocalEmbedder.

**Nguyên nhân:** Query "tối đa... mỗi ngày và mỗi tuần" retrieve Điều 106 (giờ làm việc ban đêm) thay vì Điều 105. Embedding model xử lý cả hai Điều là "về giờ làm việc" — semantic rất gần nhau. Filter `topic=thoi-gio-lam-viec` đúng chương nhưng không đủ chính xác để phân biệt Điều 105 vs 106.

**Góc nhìn 5 chiều (README):**
- **Retrieval Precision:** Top-1 không relevant, nhưng top-3 có thể chứa chunk đúng
- **Chunk Coherence:** Strategy A giữ câu tốt nhưng chunk chứa cả Điều 106 lẫn 107 vì merge — nên giờ "22 giờ" match với query "mỗi ngày"
- **Metadata Utility:** Cần `dieu_range` để filter đến Điều 105 cụ thể
- **Grounding Quality:** Nếu LLM đọc chunk Điều 106, sẽ trả lời sai về giờ ban đêm thay vì giờ bình thường
- **Data Strategy:** Extraction không giữ `\n\n` làm RecursiveChunker thất bại

**Đề xuất:** Thêm `dieu_range` vào metadata và dùng `search_with_filter({'topic':'...', 'dieu_range':'105'})` cho queries về điều khoản cụ thể.

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
