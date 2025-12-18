# Job Recommendation Service

Dịch vụ gợi ý việc làm tự động cho ứng viên dựa trên CV, sử dụng PhoBERT embedding và cosine similarity.

## Mục đích

Service này tự động phân tích CV của ứng viên và các tin tuyển dụng, sau đó gợi ý **Top K công việc phù hợp nhất** cho mỗi CV dựa trên độ tương đồng ngữ nghĩa (semantic similarity).

**Tính năng chính:**
- Tạo vector embedding cho CV và Job bằng PhoBERT (tiếng Việt)
- Tính toán độ tương đồng bằng Cosine Similarity
- Tự động cập nhật embedding khi nội dung thay đổi (content hash)
- Chạy định kỳ theo lịch (scheduler) hoặc chạy một lần

## Cấu trúc thư mục

```
Recommend-Service/
├── recommend_service/
│   ├── config/
│   │   └── settings.py          # Cấu hình (DB, model, scheduler)
│   ├── database/
│   │   ├── connection.py        # Kết nối PostgreSQL
│   │   └── repositories.py      # CRUD operations cho CV, Job, Recommendation
│   ├── models/
│   │   └── schemas.py           # Data classes (CVData, JobData)
│   ├── services/
│   │   ├── embedding.py         # Tạo embedding bằng PhoBERT
│   │   ├── similarity.py        # Tính cosine similarity
│   │   └── recommendation.py    # Pipeline chính
│   ├── scheduler/
│   │   └── jobs.py              # APScheduler jobs
│   └── main.py                  # Entry point
├── scripts/
│   ├── import_data.py           # Script import dữ liệu
│   └── check_counts.py          # Kiểm tra số lượng records
├── data/                        # Thư mục chứa data
├── schema.prisma                # Database schema (reference)
└── requirements.txt             # Dependencies
```

## Flow hoạt động

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Jobs từ Database                                  │
│  ─────────────────────────────                                  │
│  • Lấy tất cả jobs có status = ACTIVE                           │
│  • Lấy skills và requirements của mỗi job                       │
│  • Kiểm tra content_hash để xác định cần update embedding không │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Generate Job Embeddings (nếu cần)                      │
│  ─────────────────────────────────────────                      │
│  • Title → PhoBERT → title_embedding                            │
│  • Skills → PhoBERT → skills_embedding                          │
│  • Requirements → PhoBERT → requirement_embedding               │
│  • Lưu embeddings vào DB                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Load CVs từ Database                                   │
│  ────────────────────────────                                   │
│  • Lấy tất cả CVs có isMain = true                              │
│  • Lấy skills và experiences của mỗi CV                         │
│  • Kiểm tra content_hash                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Generate CV Embeddings (nếu cần)                       │
│  ────────────────────────────────────────                       │
│  • Title + Current Position → PhoBERT → title_embedding         │
│  • Skills → PhoBERT → skills_embedding                          │
│  • Experiences → PhoBERT → experience_embedding                 │
│  • Lưu embeddings vào DB                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Calculate Similarity                                   │
│  ────────────────────────────                                   │
│  Với mỗi CV:                                                    │
│    • Tính cosine_similarity(cv.title_embedding, job.title_emb)  │
│    • Sắp xếp jobs theo similarity giảm dần                      │
│    • Lấy Top K jobs (mặc định K=20)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Save Recommendations                                   │
│  ────────────────────────────                                   │
│  • Upsert vào bảng recommend_jobs_for_cv                        │
│  • Mỗi CV có tối đa K job recommendations                       │
│  • Lưu similarity score để ranking                              │
└─────────────────────────────────────────────────────────────────┘
```

## Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống

- Python 3.9+
- PostgreSQL database (đã có sẵn từ JobsConnect)
- RAM tối thiểu 4GB (để load PhoBERT model)

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

### 3. Cấu hình

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

### 4. Chạy service

```bash
# Activate virtual environment
source venv/bin/activate

# Test kết nối database
python -m recommend_service.main --mode test

# Chạy một lần (không schedule)
python -m recommend_service.main --mode once

# Chạy với scheduler (mặc định: mỗi 12 giờ)
python -m recommend_service.main --mode schedule

# Chạy scheduler nhưng không chạy ngay lập tức
python -m recommend_service.main --mode schedule --no-immediate
```

## Các mode chạy

| Mode | Mô tả |
|------|-------|
| `test` | Chỉ test kết nối database rồi thoát |
| `once` | Chạy pipeline một lần rồi thoát |
| `schedule` | Chạy pipeline ngay + lên lịch chạy định kỳ |

## Database Schema (liên quan)

Service sử dụng các bảng sau từ database JobsConnect:

**Input:**
- `cvs` - Thông tin CV (lấy các CV có `isMain = true`)
- `cv_skills` - Kỹ năng của CV
- `work_experiences` - Kinh nghiệm làm việc
- `jobs` - Tin tuyển dụng (lấy các job có `status = ACTIVE`)
- `job_skills` - Kỹ năng yêu cầu
- `job_requirements` - Yêu cầu công việc

**Output:**
- `recommend_jobs_for_cv` - Kết quả gợi ý (cv_id, job_id, similarity)

**Embedding fields (được thêm vào bảng có sẵn):**
- `cvs.titleEmbedding`, `cvs.skillsEmbedding`, `cvs.experienceEmbedding`
- `jobs.titleEmbedding`, `jobs.skillsEmbedding`, `jobs.requirementEmbedding`
- `contentHash` - Hash để kiểm tra nội dung có thay đổi không

## Logs

Service ghi log ra:
- Console (stdout)
- File `recommend_service.log`

## Mở rộng

### Thay đổi cách tính similarity

Hiện tại service chỉ sử dụng `title_embedding`. Để sử dụng weighted combination:

Sửa file `recommend_service/services/similarity.py`:

```python
def __init__(self):
    self.title_weight = 0.5      # Trọng số title
    self.skills_weight = 0.3     # Trọng số skills
    self.experience_weight = 0.2  # Trọng số experience
```

Và uncomment phần code trong method `calculate_similarity()`.

## Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `pip: command not found` | Cài Python: `sudo apt install python3-pip` |
| `ModuleNotFoundError: torch` | Chạy `pip install -r requirements.txt` |
| `Database connection failed` | Kiểm tra DATABASE_URL trong .env |
| `CUDA out of memory` | Model sẽ tự động dùng CPU nếu không có GPU |
| `No active jobs found` | Kiểm tra database có jobs với status=ACTIVE |
