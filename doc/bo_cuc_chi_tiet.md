# Bố Cục Hệ Thống Gợi Ý Việc Làm

> **Lưu ý**: Phần 0 (Giới thiệu) đã có trong file LaTeX `section_51_gioi_thieu.tex`, không cần viết lại trong markdown này.

---

## 1. CƠ SỞ LÝ THUYẾT (Section 5.2)

### 1.1. Tổng quan về Hệ thống Gợi ý (Recommendation Systems)
- Phân loại hệ thống gợi ý:
  - Collaborative Filtering
  - Content-based Filtering
  - Hybrid Approaches
- Lý do chọn Content-based approach cho bài toán gợi ý việc làm

### 1.2. Embedding và Học biểu diễn ngữ nghĩa 

#### 1.2.1. BERT - Bidirectional Encoder Representations from Transformers
- **Tại sao BERT phù hợp cho gợi ý dựa trên ý nghĩa?**
  - Tokenization: Cách BERT phân tích văn bản
  - Self-Attention Mechanism: Nắm bắt ngữ cảnh hai chiều
  - Masked Language Modeling (MLM): Pre-training strategy
  - Transfer Learning: Fine-tuning cho domain cụ thể

#### 1.2.2. PhoBERT - BERT cho tiếng Việt
- **Tại sao sử dụng PhoBERT?**
  - Đặc thù ngôn ngữ tiếng Việt
  - Benchmark so sánh với các mô hình khác (mBERT, XLM-R)
  - Kết quả thực nghiệm trên các task NLP tiếng Việt
  - Visualization: t-SNE/UMAP của embedding space
  - Dẫn chứng từ các bài báo khoa học

### 1.3. Lưu trữ và Truy vấn Vector (Vector Storage & Retrieval)

#### 1.3.1. Vector Database vs Traditional Database
- Lý do chọn PostgreSQL + FAISS:
  - vì đang dùng PostgreSQL cho hệ thống, không muốn dùng thêm dịch vụ mới
  - FAISS là thư viện mã nguồn mở, dễ tích hợp ,...

#### 1.3.2. FAISS - Facebook AI Similarity Search
- **Giới thiệu FAISS**
  - Flat Index: Brute-force exact search
  - IVF (Inverted File Index): Clustering-based approximate search
  - HNSW (Hierarchical Navigable Small World): Graph-based search
- **Lựa chọn index type IVF**
  - Trade-off giữa accuracy, speed, memory
  - Lý do chọn index type cụ thể cho hệ thống

#### 1.3.3. Similarity Metrics
- **Cosine Similarity**: Đo góc giữa hai vector
  - Công thức toán học
  - Ưu điểm: Không phụ thuộc vào magnitude
  - Khi nào nên sử dụng
- **Các metrics khác**: Euclidean distance, Dot product (sơ qua )
- So sánh và lựa chọn metric phù hợp

#### 1.3.4. Chiến lược Lưu trữ và Truy vấn
- In-memory vs Disk-based storage
- Index persistence và loading strategy
- Query optimization techniques

### 1.4. Cascade Filtering - Lọc Phân Tầng

**Định nghĩa**: Cascade filtering (lọc phân tầng) là chiến lược xử lý dữ liệu theo nhiều giai đoạn tuần tự, mỗi giai đoạn sử dụng các tiêu chí khác nhau để lọc dần tập ứng viên từ tập lớn đến tập nhỏ có chất lượng cao [1, 2].

**Nguyên lý cốt lõi**:
- Xử lý theo thứ tự từ **thô đến tinh** (coarse-to-fine)
- Sử dụng **features đơn giản trước**, features phức tạp sau
- Mỗi giai đoạn **loại bỏ dần** các ứng viên không phù hợp
- **Tối ưu chi phí tính toán** bằng cách chỉ áp dụng tính toán phức tạp cho tập nhỏ

**Cơ sở lý thuyết**: ( trình bày tổng quan về các nghiên cứu nền tảng )

**Áp dụng trong hệ thống**:
```
14,634 Jobs
  ↓ Round 1 (Title - FAISS): O(log N)
1,000 candidates
  ↓ Round 2 (Experience - Loop): O(K₁)
100 candidates
  ↓ Round 3 (Skills - Loop): O(K₂)
10 final jobs
```

**Ưu điểm**: Speedup 40x, latency thấp (18-33ms/CV), memory efficient (52 MB FAISS index)

**Nhược điểm**: Có thể miss jobs nếu title similarity thấp, cần tune K values carefully

**Chi tiết thuật toán**: Xem Section 2.4.2 (Cascade Filtering Algorithm)

**Tài liệu Tham khảo**:

[1] Covington, P., Adams, J., & Sargin, E. (2016). "Deep Neural Networks for YouTube Recommendations". *RecSys*.

[2] Viola, P., & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features". *CVPR*.

[3] Blum, A., & Mitchell, T. (1998). "Combining Labeled and Unlabeled Data with Co-Training". *COLT*.

[4] Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale Similarity Search with GPUs". *IEEE Transactions on Big Data*.

[5] Yi, X., et al. (2019). "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations". *RecSys*.

---

## 2. THIẾT KẾ VÀ CÀI ĐẶT HỆ THỐNG (Section 5.3)

### 2.1. Kiến trúc Tổng thể (System Architecture)
- Architecture diagram
- Data flow diagram
- Component interaction

### 2.2. Embedding Strategy

**Các trường được Embedding**:
- **Job**: `title_embedding`, `skills_embedding`, `requirement_embedding`
- **CV**: `title_embedding`, `skills_embedding`, `experience_embedding`
- **Vector dimension**: 768 (PhoBERT base)

**Lý do embedding riêng từng trường**:
- Cascade filtering sử dụng từng embedding cho mỗi round
- Kiểm soát được trọng số và tính similarity có hướng
- Trade-off: Tăng storage (~3KB/job, ~3KB/CV) nhưng tăng accuracy

### 2.3. Lưu trữ Dữ liệu (Data Storage)

**PostgreSQL Schema**:
- **Job table**: Lưu `title_embedding`, `skills_embedding`, `requirement_embedding`, `content_hash`
- **CV table**: Lưu `title_embedding`, `skills_embedding`, `experience_embedding`, `content_hash`
- **Recommendation table**: `RecommendJobForCV` (cv_id, job_id, similarity_score, created_at)
- **Type**: `bytea` (serialize array)

**FAISS Index**:
- **Build**: Từ job title embeddings trong PostgreSQL → FAISS IVFFlat index (nlist=100, nprobe=10)
- **Storage**: Save to disk (`.faiss` files), load to RAM khi query
- **Memory**: ~52 MB cho 14,634 jobs

**Chiến lược lưu trữ**: Dual storage
- PostgreSQL: Source of truth, persistent storage
- FAISS: In-memory query optimization layer

### 2.4. Luồng Truy vấn (Query Pipeline)

#### 2.4.1. CV-Job Recommendation Flow
```
1. Load CVs and Jobs from PostgreSQL
2. Generate/update embeddings (nếu content thay đổi, check via content_hash)
3. Build/load FAISS index từ job title embeddings
4. Cascade filtering (3 rounds) → Top 10 jobs per CV
5. Save recommendations to RecommendJobForCV table
```

#### 2.4.2. Cascade Filtering Algorithm - Thuật toán Lọc Phân Tầng

**Tổng quan**: Cascade filtering sử dụng ba vòng lọc tuần tự để tăng độ chính xác:

```
14,634 Jobs → [Round 1: Title-FAISS] → 1,000 jobs → [Round 2: Experience] → 100 jobs → [Round 3: Skills] → 10 final jobs
```

**Các vòng lọc**:

**Round 1 - Title Filtering (FAISS)**:
- **Input**: CV title embedding, FAISS index từ job titles
- **Method**: FAISS IVFFlat search, cosine similarity
- **Output**: Top 1,000 jobs
- **Time**: 1-3ms, O(log N)
- **Lý do dùng FAISS**: Search space lớn (14K jobs), cần approximate nearest neighbor

**Round 2 - Experience-Requirements Filtering**:
- **Input**: 1,000 jobs từ Round 1
- **Method**: Python loop, cosine similarity giữa `cv.experience_embedding` và `job.requirement_embedding`
- **Output**: Top 100 jobs
- **Time**: 15-25ms, O(K₁)
- **Lý do dùng loop**: Chỉ 1,000 items, overhead building FAISS index > direct computation
- **Xử lý missing embeddings**: CV fresher → score=0.3, Job no requirements → score=0.5

**Round 3 - Skills Filtering**:
- **Input**: 100 jobs từ Round 2
- **Method**: Python loop, cosine similarity giữa `cv.skills_embedding` và `job.skills_embedding`
- **Output**: 10 final jobs
- **Time**: 2-5ms, O(K₂)
- **Xử lý missing embeddings**: CV no skills → score=0.2, Job no skills → score=0.4



### 2.5. Tối ưu hóa

**Content Hash Cache**:
- Lưu `content_hash` trong Job/CV tables
- Chỉ re-embed khi content thay đổi
- chỉ cử lý job active ,...

**FAISS Index Update**:
- Rebuild index khi có jobs mới (threshold-based hoặc time-based)
- Save index to disk để tránh rebuild sau restart

---

## 3. ĐÁNH GIÁ VÀ THỰC NGHIỆM (Section 5.4)

### 3.1. Cơ sở Đánh giá

#### 3.1.1. Ground Truth Dataset
- **Phương pháp tạo dữ liệu**:
  - Synthetic data: Ghép keywords giữa CV và Job
  - Rule-based matching: Một CV khớp với ít nhất một Job
  - Số lượng: 5000 CV-Job pairs
- **Đảm bảo chất lượng**:
  - Manual verification (sample)
  - Diversity: Đa dạng ngành nghề, level

#### 3.1.2. Baseline Methods
- Random recommendation
- TF-IDF + Cosine similarity
- Word2Vec/FastText embeddings
- Rule-based matching

### 3.2. Metrics Đánh giá

#### 3.2.1. Ranking Metrics

**1. Pass@K (Recall@K)**
- **Định nghĩa**:
  - Với mỗi CV, gợi ý Top K jobs
  - Pass nếu ground truth job nằm trong Top K
  - Pass@K = (Số CVs pass) / (Tổng số CVs)
- **K = 10**: Standard cho hệ thống gợi ý việc làm
- **Weighted variant**:
  - Position-aware: Rank cao → weight lớn hơn
  - Formula: `score = Σ(1/rank) nếu trong Top K, else 0`

**2. HR@K (Hit Ratio @ K)**
- **Công thức**: `HR@K = (Số lượng hits trong top K) / (Tổng số test cases)`
- So sánh với Pass@K

**3. NDCG@K (Normalized Discounted Cumulative Gain)**
- **Công thức**:
  ```
  DCG@K = Σ (rel_i / log2(i+1)) for i in 1..K
  NDCG@K = DCG@K / IDCG@K
  ```
- **Ưu điểm**: Xem xét vị trí ranking
- **Threshold**: NDCG > 0.7 là tốt

**4. MRR (Mean Reciprocal Rank)**
- **Công thức**: `MRR = Average(1/rank_of_first_relevant_item)`
- Đánh giá vị trí của kết quả đầu tiên relevant

**5. Precision@K và Recall@K**
- **Precision@K**: Tỷ lệ relevant items trong Top K
- **Recall@K**: Tỷ lệ relevant items tìm được / tổng relevant items
- **F1@K**: Harmonic mean của Precision và Recall

#### 3.2.2. Business Metrics
- **Click-Through Rate (CTR)**: % jobs được click
- **Apply Rate**: % jobs được ứng tuyển
- **Diversity**: Đa dạng của recommendations
- **Coverage**: % jobs được gợi ý ít nhất 1 lần

### 3.3. Kết quả Thực nghiệm

#### 3.3.1. Comparison Table
| Method | Pass@10 | HR@10 | NDCG@10 | MRR | Training Time | Query Time |
|--------|---------|-------|---------|-----|---------------|------------|
| Random | - | - | - | - | - | - |
| TF-IDF | - | - | - | - | - | - |
| PhoBERT (single field) | - | - | - | - | - | - |
| PhoBERT (multi-field) | - | - | - | - | - | - |
| PhoBERT + FAISS | - | - | - | - | - | - |

#### 3.3.2. Ablation Study
- **Impact của từng trường embedding**:
  - Title only
  - Title + Description
  - All fields
- **Impact của weights**:
  - Uniform weights
  - Learned weights
- **Impact của filtering rounds**:
  - No filter
  - 1 filter
  - 3 filters (full pipeline)

#### 3.3.3. Visualization
- **Embedding Space Visualization**:
  - t-SNE plot: Jobs và CVs trong 2D space
  - Cluster visualization: Ngành nghề khác nhau
- **Performance Charts**:
  - NDCG@K cho K = 1, 5, 10, 20, 50
  - Precision-Recall curve
- **Error Analysis**:
  - False positives examples
  - False negatives examples
  - Common failure patterns

### 3.4. Performance Benchmarks

#### 3.4.1. Computational Performance
- **Embedding time**:
  - Single job: X ms
  - Batch (100 jobs): Y ms
- **Index building time**:
  - 10K jobs: A seconds
  - 100K jobs: B seconds
- **Query latency**:
  - FAISS search: X ms
  - Full pipeline (với filters): Y ms
  - P50, P95, P99 latency

#### 3.4.2. Resource Usage
- **Memory**:
  - FAISS index size: X MB
  - Peak RAM usage: Y GB
- **Storage**:
  - Vector storage in PostgreSQL: X GB
  - Index files: Y MB
- **CPU**:
  - Embedding: X% utilization
  - Query: Y% utilization

