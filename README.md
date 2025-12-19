# Job Recommendation Service

Dịch vụ gợi ý việc làm tự động cho ứng viên và tìm kiếm công việc tương tự, sử dụng PhoBERT embedding, FAISS và cosine similarity.

## Tính năng chính

**1. CV-Job Recommendations (Gợi ý việc làm cho CV)**
- Phân tích CV của ứng viên và gợi ý **Top K công việc phù hợp nhất**
- Sử dụng PhoBERT để tạo vector embedding cho CV và Job
- Tính độ tương đồng bằng Cosine Similarity
- Tìm kiếm nhanh với FAISS index (IndexIVFFlat + K-means clustering)

**2. Similar Jobs (Tìm công việc tương tự)**
- Tính toán Top K công việc tương tự cho mỗi công việc
- Giúp đề xuất các vị trí liên quan khi ứng viên xem chi tiết job
- Sử dụng chung FAISS index với CV-Job Recommendations

**3. Content Hash Caching (Smart Embedding Update)**
- Tự động detect thay đổi nội dung bằng MD5 hash
- Chỉ tạo lại embedding khi CV/Job thay đổi
- Skip embedding nếu nội dung không đổi
- Tiết kiệm ~90% thời gian khi re-run

**4. Performance tối ưu**
- FAISS với K-means clustering (IVFFlat): nhanh hơn 10-50x so với brute force
- Batch processing với vectorized operations
- Shared index: tiết kiệm 50% RAM và disk space

## Kiến trúc hệ thống

```
┌──────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION PIPELINE                        │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────────┐              ┌───────────────────────┐
│ Step 1: CV-Job Recs   │              │ Step 2: Similar Jobs  │
│ ─────────────────────│              │ ─────────────────────│
│ • Load ACTIVE jobs    │              │ • Load ACTIVE jobs    │
│ • Embed jobs (cache)  │              │ • Reuse embeddings    │
│ • Embed CVs (cache)   │              │ • Load shared index   │
│ • Build FAISS index   │              │ • Calculate similar   │
│ • Save shared index   │◄─────────────┤ • Save to DB          │
│ • Calculate CV→Job    │   shared     │                       │
│ • Save to DB          │   index      │                       │
└───────────────────────┘              └───────────────────────┘
         │                                          │
         └────────────┬─────────────────────────────┘
                      ▼
        ./faiss_data/shared_jobs.faiss
        (Shared K-means Index)
```

## Cấu trúc thư mục

```
Recommend-Service/
├── recommend_service/
│   ├── config/
│   │   └── settings.py                    # Cấu hình (DB, model, scheduler)
│   ├── database/
│   │   ├── connection.py                  # Kết nối PostgreSQL
│   │   └── repositories.py                # CRUD operations
│   ├── models/
│   │   └── schemas.py                     # Data classes (CVData, JobData)
│   ├── services/
│   │   ├── embedding.py                   # Tạo embedding bằng PhoBERT
│   │   ├── similarity.py                  # CV-Job similarity (FAISS)
│   │   ├── similar_jobs.py                # Job-Job similarity (FAISS)
│   │   ├── recommendation.py              # Pipeline CV-Job Recommendations
│   │   └── similar_jobs_recommendation.py # Pipeline Similar Jobs
│   ├── scheduler/
│   │   └── jobs.py                        # APScheduler jobs
│   └── main.py                            # Entry point
├── scripts/
│   ├── import_data.py                     # Import dữ liệu
│   ├── calculate_similar_jobs.py          # Chạy Similar Jobs standalone
│   ├── query_similar_jobs.py              # Query similar jobs cho 1 job
│   └── run_full_pipeline.py               # Chạy cả 2 services
├── faiss_data/
│   └── shared_jobs.faiss                  # Shared FAISS index + metadata
├── data/                                  # Thư mục chứa data
├── schema.prisma                          # Database schema (reference)
├── requirements.txt                       # Dependencies
├── README.md                              # Documentation này
└── SHARED_FAISS_INDEX.md                  # Chi tiết về shared index
```

## Flow hoạt động chi tiết

### Step 1: CV-Job Recommendations

```
1. Load ACTIVE Jobs từ DB
   └─> Lấy jobs.status = ACTIVE
   └─> Lấy job_skills, job_requirements

2. Generate Job Embeddings (nếu content_hash thay đổi)
   └─> title → PhoBERT → title_embedding (768-dim)
   └─> skills → PhoBERT → skills_embedding
   └─> requirements → PhoBERT → requirement_embedding
   └─> Save to jobs table

3. Build/Load FAISS Index
   └─> Try load existing shared index
   └─> If not exist: Build IVFFlat index (K-means clustering)
   └─> nlist=100 clusters, nprobe=10
   └─> Save to ./faiss_data/shared_jobs.faiss

4. Load CVs từ DB
   └─> Lấy cvs.isMain = true
   └─> Lấy cv_skills, work_experiences

5. Generate CV Embeddings (nếu content_hash thay đổi)
   └─> title + currentPosition → PhoBERT → title_embedding
   └─> skills → PhoBERT → skills_embedding
   └─> experiences → PhoBERT → experience_embedding
   └─> Save to cvs table

6. Calculate Similarity (FAISS)
   └─> Normalize CV embedding (L2 norm)
   └─> Search FAISS index: find Top K jobs
   └─> Distance = Inner Product (normalized = cosine similarity)

7. Save Recommendations
   └─> Upsert to recommend_jobs_for_cv
   └─> Fields: cvId, jobId, similarity
```

### Step 2: Similar Jobs

```
1. Load ACTIVE Jobs từ DB
   └─> Lấy jobs.status = ACTIVE (same as Step 1)
   └─> Lấy job_skills, job_requirements

2. Check Embeddings
   └─> Embeddings already exist from Step 1 (content_hash match)
   └─> Skip re-embedding

3. Load Shared FAISS Index
   └─> Load from ./faiss_data/shared_jobs.faiss
   └─> Saved by Step 1
   └─> If not exist: Build new index

4. Calculate Similar Jobs (FAISS Batch)
   └─> For each job:
   │   └─> Normalize job embedding
   │   └─> Search FAISS index: find Top K+1 similar jobs
   │   └─> Exclude self-match
   └─> Batch process with vectorized operations

5. Save Similar Jobs
   └─> Batch upsert to similar_jobs
   └─> Fields: jobId, similarJobId, similarity
```

## Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống

- **Python 3.9+**
- **PostgreSQL database** (đã có sẵn từ JobsConnect)
- **RAM tối thiểu 4GB** (để load PhoBERT model)
- **Disk space:** ~2GB cho model + FAISS index

### 2. Cài đặt trên Ubuntu/Debian

```bash
# Clone repository
git clone https://github.com/Duy-Thong/Recommend-Service
cd Recommend-Service

# Cài Python và pip (nếu chưa có)
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

### 3. Cài đặt trên Windows

```bash
# Clone repository
git clone https://github.com/Duy-Thong/Recommend-Service
cd Recommend-Service

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt
```

### 4. Cấu hình

Tạo file `.env` trong thư mục gốc:

```bash
# Database connection (bắt buộc)
DATABASE_URL=postgresql://user:password@host:5432/jobsconnect

# Embedding model (tùy chọn, mặc định là PhoBERT)
EMBEDDING_MODEL=VoVanPhuc/sup-SimCSE-VietNamese-phobert-base

# Số lượng jobs gợi ý cho mỗi CV (mặc định: 20)
TOP_K_JOBS=20

# Chu kỳ chạy scheduler, tính bằng giờ (mặc định: 12)
SCHEDULE_INTERVAL_HOURS=12

# Batch size khi xử lý (mặc định: 100)
BATCH_SIZE=100
```

## Hướng dẫn sử dụng

### Chạy Full Pipeline (Recommended)

Chạy cả 2 services theo thứ tự tối ưu:

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Chạy full pipeline (CV-Job Recs → Similar Jobs)
python scripts/run_full_pipeline.py

# Chạy theo thứ tự ngược lại (nếu cần)
python scripts/run_full_pipeline.py --reverse
```

**Output:**
```
======================================================================
FULL RECOMMENDATION PIPELINE
======================================================================
Database: postgresql://user@host:5432/jobsconnect
Embedding model: VoVanPhuc/sup-SimCSE-VietNamese-phobert-base
Top K jobs: 20
Order: CV-Job Recommendations → Similar Jobs (recommended)
======================================================================

======================================================================
STEP 1: GENERATE CV-JOB RECOMMENDATIONS
======================================================================
[INFO] Loading jobs and CVs...
[INFO] Building FAISS index with 100 clusters (k-means)
[INFO] Saved FAISS index to ./faiss_data/shared_jobs.faiss
[INFO] Completed batch calculation for 1500 CVs
======================================================================

======================================================================
STEP 2: CALCULATE SIMILAR JOBS
======================================================================
[INFO] Attempting to load shared FAISS index from previous run
[INFO] Successfully loaded existing shared FAISS index
[INFO] Calculating similar jobs for 250 jobs
[INFO] Completed batch calculation for 250 jobs
======================================================================

Summary:
----------------------------------------------------------------------
Step 1 - CV-Job Recommendations:
  - CVs processed: 1500
  - CVs embedded: 1500
  - Jobs processed: 250
  - Jobs embedded: 250
  - Recommendations created: 30000

Step 2 - Similar Jobs:
  - Jobs processed: 250
  - Jobs embedded: 0 (reused from Step 1)
  - Similar jobs created: 2500
======================================================================
```

### Chạy từng service riêng lẻ

#### CV-Job Recommendations

```bash
# Test kết nối database
python -m recommend_service.main --mode test

# Chạy một lần
python -m recommend_service.main --mode once

# Chạy với scheduler (mỗi 12 giờ)
python -m recommend_service.main --mode schedule

# Chạy scheduler nhưng không chạy ngay
python -m recommend_service.main --mode schedule --no-immediate
```

#### Similar Jobs

```bash
# Chạy calculate similar jobs
python scripts/calculate_similar_jobs.py

# Query similar jobs cho 1 job cụ thể
python scripts/query_similar_jobs.py <job_id>
```

### Import dữ liệu test

```bash
# Import CVs và Jobs từ file JSON
python scripts/import_data.py
```

## Các mode chạy

| Mode | Mô tả | Use Case |
|------|-------|----------|
| `test` | Test kết nối database rồi thoát | Kiểm tra cấu hình |
| `once` | Chạy pipeline một lần rồi thoát | Manual trigger, testing |
| `schedule` | Chạy ngay + lên lịch chạy định kỳ | Production (chạy mỗi 12h) |

## Database Schema

### Input Tables

**CVs:**
- `cvs` - CV chính (lấy `isMain = true`)
- `cv_skills` - Kỹ năng của CV
- `work_experiences` - Kinh nghiệm làm việc

**Jobs:**
- `jobs` - Tin tuyển dụng (lấy `status = ACTIVE`)
- `job_skills` - Kỹ năng yêu cầu
- `job_requirements` - Yêu cầu công việc

### Output Tables

**Recommendations:**
```sql
-- CV-Job Recommendations
CREATE TABLE recommend_jobs_for_cv (
  id         TEXT PRIMARY KEY,
  cvId       TEXT NOT NULL,
  jobId      TEXT NOT NULL,
  similarity FLOAT NOT NULL,
  createdAt  TIMESTAMP DEFAULT NOW(),
  updatedAt  TIMESTAMP DEFAULT NOW(),
  UNIQUE(cvId, jobId)
);

-- Similar Jobs
CREATE TABLE similar_jobs (
  id           TEXT PRIMARY KEY,
  jobId        TEXT NOT NULL,
  similarJobId TEXT NOT NULL,
  similarity   FLOAT NOT NULL,
  createdAt    TIMESTAMP DEFAULT NOW(),
  updatedAt    TIMESTAMP DEFAULT NOW(),
  UNIQUE(jobId, similarJobId)
);
```

### Embedding Fields

**Thêm vào bảng có sẵn:**
```sql
-- CVs table
ALTER TABLE cvs ADD COLUMN titleEmbedding JSON;
ALTER TABLE cvs ADD COLUMN skillsEmbedding JSON;
ALTER TABLE cvs ADD COLUMN experienceEmbedding JSON;
ALTER TABLE cvs ADD COLUMN contentHash TEXT;

-- Jobs table
ALTER TABLE jobs ADD COLUMN titleEmbedding JSON;
ALTER TABLE jobs ADD COLUMN skillsEmbedding JSON;
ALTER TABLE jobs ADD COLUMN requirementEmbedding JSON;
ALTER TABLE jobs ADD COLUMN contentHash TEXT;
```

## FAISS Index

### Shared Index Architecture

Service sử dụng **1 shared FAISS index** cho cả 2 services:
- **Path:** `./faiss_data/shared_jobs.faiss`
- **Type:** IndexIVFFlat (Inverted File with K-means)
- **Metric:** Inner Product (normalized vectors = cosine similarity)

**Tham số:**
- `nlist = 100` - Số clusters cho K-means
- `nprobe = 10` - Số clusters tìm kiếm khi query
- `dimension = 768` - PhoBERT embedding size

**Lợi ích:**
- 50% giảm disk usage (1 index thay vì 2)
- 50% giảm RAM usage khi load
- Consistency - cả 2 services dùng cùng dataset (ACTIVE jobs)
- 10-50x nhanh hơn brute force

Chi tiết: xem [SHARED_FAISS_INDEX.md](SHARED_FAISS_INDEX.md)

## Content Hash Caching Mechanism

### Cơ chế hoạt động

Service sử dụng **MD5 content hash** để phát hiện khi nội dung CV/Job thay đổi:

```python
# Ví dụ: Tính content hash cho Job
import hashlib

def compute_content_hash(job_data, skills, requirements):
    """
    Tạo MD5 hash từ tất cả nội dung liên quan đến embedding
    """
    content_parts = [
        job_data.get("title", ""),
        # Skills
        " ".join([s.get("skillName", "") for s in skills]),
        # Requirements
        " ".join([f"{r.get('title', '')} {r.get('description', '')}"
                  for r in requirements])
    ]

    combined = "|".join(content_parts)
    return hashlib.md5(combined.encode()).hexdigest()
```

### Flow cập nhật embedding

```
┌─────────────────────────────────────────────────────────────┐
│ Load CV/Job from Database                                   │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Tính content hash mới từ:                                   │
│ • Job: title + skills + requirements                        │
│ • CV:  title + currentPosition + skills + experiences       │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ So sánh với hash cũ     │
         │ (từ DB: contentHash)    │
         └─────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         ▼                            ▼
    ┌─────────┐                ┌──────────────┐
    │ KHÁC    │                │ GIỐNG        │
    └─────────┘                └──────────────┘
         │                            │
         ▼                            ▼
┌──────────────────┐         ┌──────────────────┐
│ Generate embedding│         │ Skip embedding   │
│ • Call PhoBERT    │         │ • Reuse existing │
│ • Save to DB      │         │ • No API call    │
│ • Update hash     │         │ ⚡ Fast!         │
└──────────────────┘         └──────────────────┘
```

### Ví dụ thực tế

**Lần chạy đầu tiên (chưa có embedding):**
```
Job #1: "Python Developer"
├─ contentHash: null
├─ titleEmbedding: null
└─ Action: Generate embedding ⏱️ ~500ms
    ├─ Call PhoBERT API
    ├─ Save embedding to DB
    └─ Save hash: "a3f2c8d9e1b4..."
```

**Lần chạy thứ 2 (nội dung không đổi):**
```
Job #1: "Python Developer" (same content)
├─ contentHash: "a3f2c8d9e1b4..."
├─ titleEmbedding: [0.123, -0.456, ...]
└─ Action: Skip embedding ⚡ ~0ms
    └─ Hash match → reuse existing
```

**Lần chạy thứ 3 (nội dung thay đổi):**
```
Job #1: "Senior Python Developer" (title changed)
├─ contentHash: "a3f2c8d9e1b4..." (old)
├─ New hash: "b7e9d2a1f3c8..." (different)
└─ Action: Re-generate embedding ⏱️ ~500ms
    ├─ Content changed detected
    ├─ Call PhoBERT API
    ├─ Update embedding
    └─ Update hash: "b7e9d2a1f3c8..."
```

### Performance impact

**Scenario: 1000 Jobs, re-run sau 1 ngày**

| Trường hợp | Jobs thay đổi | Time without cache | Time with cache | Savings |
|------------|---------------|-------------------|----------------|---------|
| Ít update  | 10 jobs (1%)  | ~8 minutes        | ~0.8 minutes   | **90%** |
| Vừa phải   | 100 jobs (10%)| ~8 minutes        | ~1.5 minutes   | **81%** |
| Nhiều update| 500 jobs (50%)| ~8 minutes        | ~4 minutes     | **50%** |

### Database schema

```sql
-- Jobs table
ALTER TABLE jobs ADD COLUMN contentHash TEXT;
ALTER TABLE jobs ADD COLUMN titleEmbedding JSON;
ALTER TABLE jobs ADD COLUMN skillsEmbedding JSON;
ALTER TABLE jobs ADD COLUMN requirementEmbedding JSON;

-- CVs table
ALTER TABLE cvs ADD COLUMN contentHash TEXT;
ALTER TABLE cvs ADD COLUMN titleEmbedding JSON;
ALTER TABLE cvs ADD COLUMN skillsEmbedding JSON;
ALTER TABLE cvs ADD COLUMN experienceEmbedding JSON;
```

### Code implementation

**File:** `recommend_service/services/recommendation.py`

```python
def _load_and_embed_jobs(self) -> List[JobData]:
    """Load active jobs and generate embeddings if needed"""
    jobs_data = []
    raw_jobs = self.job_repo.get_active_jobs()

    for raw_job in raw_jobs:
        job_id = raw_job["id"]

        # Get related data
        skills = self.job_repo.get_job_skills(job_id)
        requirements = self.job_repo.get_job_requirements(job_id)

        # Create JobData object
        job = JobData.from_db_row(raw_job, skills, requirements)

        # Compute current hash from content
        current_hash = JobRepository.compute_content_hash(
            raw_job, skills, requirements
        )

        # Check if embedding needs update
        needs_update = (
            job.content_hash != current_hash or  # Hash changed
            not job.title_embedding               # No embedding yet
        )

        if needs_update:
            logger.info(f"Generating embeddings for job: {job_id}")
            self._generate_job_embeddings(job, skills, requirements, current_hash)
        else:
            logger.debug(f"Skipping job {job_id} - content unchanged")

        jobs_data.append(job)

    return jobs_data
```

### Lợi ích

1. **Tiết kiệm thời gian:**
   - Không cần re-embed nếu nội dung không đổi
   - ~90% faster khi re-run với ít thay đổi

2. **Tiết kiệm chi phí:**
   - Giảm API calls đến PhoBERT
   - Giảm computational cost

3. **Consistency:**
   - Chỉ update khi thực sự cần
   - Embedding luôn sync với content

4. **Scalability:**
   - Càng nhiều data, càng thấy lợi ích
   - Production-ready cho hàng triệu records

## Performance

### Benchmark (1500 CVs × 250 Jobs)

| Method | Time | Speed |
|--------|------|-------|
| **Brute Force** | ~45 minutes | 1x |
| **FAISS IVFFlat** | ~3 minutes | **15x faster** |

### Scaling

- **10,000 CVs × 1,000 Jobs:** ~15 minutes với FAISS
- **100,000 CVs × 5,000 Jobs:** ~2 hours với FAISS
- RAM usage: ~500MB cho 5000 jobs index

## Logs

Service ghi log ra:
- **Console** (stdout) - Real-time monitoring
- **File** `recommend_service.log` - Persistent logs

**Log levels:**
- `INFO` - Pipeline progress, statistics
- `WARNING` - Missing embeddings, empty results
- `ERROR` - Database errors, exceptions

## Mở rộng và tùy chỉnh

### 1. Thay đổi cách tính similarity

Hiện tại chỉ dùng `title_embedding`. Để dùng weighted combination:

**File:** `recommend_service/services/similarity.py`

```python
def __init__(self):
    # Uncomment và điều chỉnh weights
    self.title_weight = 0.5      # Trọng số title
    self.skills_weight = 0.3     # Trọng số skills
    self.experience_weight = 0.2  # Trọng số experience
```

Sau đó uncomment code trong method `calculate_similarity()`.

### 2. Thay đổi Top K

**File:** `.env`
```bash
TOP_K_JOBS=30  # Thay đổi từ 20 sang 30
```

### 3. Thay đổi FAISS parameters

**File:** `recommend_service/services/recommendation.py` và `similar_jobs_recommendation.py`

```python
self.similarity_service = SimilarityService(
    index_path=shared_index_path,
    index_type="IVFFlat",  # Hoặc "Flat" cho brute force
    nlist=200,             # Tăng số clusters
    nprobe=20              # Tăng số clusters search
)
```

**Trade-off:**
- `nlist` ↑ → build time ↑, search time ↓
- `nprobe` ↑ → accuracy ↑, search time ↑

### 4. Thay đổi embedding model

**File:** `.env`
```bash
# Dùng model khác từ HuggingFace
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Troubleshooting

### Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| `pip: command not found` | Chưa cài Python/pip | `sudo apt install python3-pip` |
| `ModuleNotFoundError: torch` | Chưa cài dependencies | `pip install -r requirements.txt` |
| `Database connection failed` | Sai DATABASE_URL | Kiểm tra .env, test connection |
| `CUDA out of memory` | GPU không đủ RAM | Model tự động dùng CPU |
| `No active jobs found` | DB không có jobs ACTIVE | Thêm jobs hoặc check status |
| `FAISS index not found` | Chưa chạy Step 1 | Chạy CV-Job Recs trước |
| `Index dimension mismatch` | Model thay đổi | Xóa index, rebuild |

### Debug commands

```bash
# Test database connection
python -m recommend_service.main --mode test

# Check số lượng records
python scripts/check_counts.py

# Rebuild FAISS index
rm -rf faiss_data/
python scripts/run_full_pipeline.py
```

### Performance issues

**Nếu service chạy chậm:**

1. **Kiểm tra FAISS đang được dùng:**
   - Check logs: `"Building FAISS index"` hoặc `"Loading FAISS index"`
   - Nếu không thấy → set `use_faiss=True`

2. **Tăng batch size:**
   ```bash
   BATCH_SIZE=500  # trong .env
   ```

3. **Tối ưu FAISS params:**
   ```python
   nlist=50   # Giảm clusters nếu ít jobs
   nprobe=5   # Giảm search clusters
   ```

4. **Sử dụng GPU (nếu có):**
   ```bash
   pip uninstall faiss-cpu
   pip install faiss-gpu
   ```

## API Integration (Future)

Service này có thể expose API endpoints:

```python
# GET /api/recommendations/:cvId
# Trả về Top K jobs cho CV

# GET /api/similar-jobs/:jobId
# Trả về Top K similar jobs
```

Implementation: Thêm FastAPI/Flask wrapper cho services.

## Contributing

Contributions are welcome! Vui lòng:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## License

MIT License - xem file LICENSE

## Contact

- **Author:** Dao Duy Thong
- **Repository:** https://github.com/Duy-Thong/Recommend-Service
- **Issues:** https://github.com/Duy-Thong/Recommend-Service/issues

---

**Last updated:** 2025-12-19
