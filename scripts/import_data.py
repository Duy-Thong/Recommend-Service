"""
Script to import CSV data into the database.
Maps CSV columns to database schema and creates nested records (skills, experiences, etc.)
"""

import csv
import logging
import random
import sys
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommend_service.database.connection import DatabaseConnection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================
# ENUM MAPPINGS
# ============================================

GENDER_MAP = {
    "nam": "MALE",
    "nữ": "FEMALE",
    "khác": "OTHER",
    "male": "MALE",
    "female": "FEMALE",
    "other": "OTHER",
}

EXPERIENCE_LEVEL_MAP = {
    "thực tập sinh": "INTERN",
    "intern": "INTERN",
    "fresher": "FRESHER",
    "nhân viên": "JUNIOR",
    "junior": "JUNIOR",
    "middle": "MIDDLE",
    "senior": "SENIOR",
    "trưởng nhóm": "LEAD",
    "lead": "LEAD",
    "quản lý": "MANAGER",
    "manager": "MANAGER",
    "không yêu cầu kinh nghiệm": "FRESHER",
    "chưa có kinh nghiệm": "FRESHER",
    "dưới 1 năm": "FRESHER",
    "1-3 năm": "JUNIOR",
    "3-5 năm": "MIDDLE",
    "5-10 năm": "SENIOR",
    "trên 10 năm": "LEAD",
}

JOB_TYPE_MAP = {
    "full time": "FULL_TIME",
    "full-time": "FULL_TIME",
    "toàn thời gian": "FULL_TIME",
    "part time": "PART_TIME",
    "part-time": "PART_TIME",
    "bán thời gian": "PART_TIME",
    "contract": "CONTRACT",
    "hợp đồng": "CONTRACT",
    "internship": "INTERNSHIP",
    "thực tập": "INTERNSHIP",
    "freelance": "FREELANCE",
}

COMPANY_SIZE_MAP = {
    "1-9": "STARTUP",
    "10-24": "SMALL",
    "25-99": "SMALL",
    "100-499": "MEDIUM",
    "500-999": "LARGE",
    "1000+": "ENTERPRISE",
    "trên 1000": "ENTERPRISE",
}

SKILL_LEVEL_MAP = {
    "beginner": "BEGINNER",
    "intermediate": "INTERMEDIATE",
    "advanced": "ADVANCED",
    "expert": "EXPERT",
}

# Common skills to extract from job descriptions/requirements
COMMON_SKILLS = [
    # ============ IT / TECH ============
    # Programming Languages
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C#",
    "C++",
    "PHP",
    "Ruby",
    "Go",
    "Golang",
    "Swift",
    "Kotlin",
    "Rust",
    "Scala",
    "R",
    "MATLAB",
    "Perl",
    "Shell",
    "Bash",
    # Web Frontend
    "HTML",
    "CSS",
    "React",
    "ReactJS",
    "React.js",
    "Angular",
    "AngularJS",
    "Vue",
    "Vue.js",
    "VueJS",
    "Next.js",
    "NextJS",
    "Nuxt.js",
    "jQuery",
    "Bootstrap",
    "Tailwind",
    "SASS",
    "SCSS",
    "Webpack",
    # Web Backend
    "Node.js",
    "NodeJS",
    "Express",
    "Express.js",
    "Django",
    "Flask",
    "FastAPI",
    "Spring",
    "Spring Boot",
    "Laravel",
    "Symfony",
    "Rails",
    "Ruby on Rails",
    "ASP.NET",
    ".NET",
    ".NET Core",
    # Mobile
    "React Native",
    "Flutter",
    "iOS",
    "Android",
    "Xamarin",
    # Databases
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Redis",
    "Elasticsearch",
    "Oracle",
    "SQL Server",
    "SQLite",
    "Cassandra",
    "DynamoDB",
    "Firebase",
    # Cloud & DevOps
    "AWS",
    "Azure",
    "GCP",
    "Google Cloud",
    "Docker",
    "Kubernetes",
    "K8s",
    "Jenkins",
    "CI/CD",
    "Terraform",
    "Ansible",
    "Linux",
    "Unix",
    "Nginx",
    "Apache",
    # Data & AI
    "Machine Learning",
    "Deep Learning",
    "AI",
    "TensorFlow",
    "PyTorch",
    "Keras",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "NLP",
    "Computer Vision",
    "Big Data",
    "Hadoop",
    "Spark",
    "Data Science",
    # Tools
    "Git",
    "GitHub",
    "GitLab",
    "Bitbucket",
    "Jira",
    "Confluence",
    "Agile",
    "Scrum",
    "REST API",
    "RESTful",
    "GraphQL",
    "Microservices",
    "API",
    # Design
    "Figma",
    "Adobe XD",
    "Photoshop",
    "Illustrator",
    "UI/UX",
    "Webflow",
    "Shopify",
    "WordPress",
    "InDesign",
    "After Effects",
    "Premiere",
    "CorelDraw",
    "Canva",
    "Sketch",
    # ============ KINH DOANH / BÁN HÀNG ============
    "Bán hàng",
    "Sales",
    "Telesales",
    "Tư vấn bán hàng",
    "Chăm sóc khách hàng",
    "CSKH",
    "B2B",
    "B2C",
    "KPI",
    "Đàm phán",
    "Thương lượng",
    "Chốt sales",
    "Tìm kiếm khách hàng",
    "Quản lý khách hàng",
    "CRM",
    "Salesforce",
    "HubSpot",
    "Phát triển thị trường",
    # ============ MARKETING / TRUYỀN THÔNG ============
    "Marketing",
    "Digital Marketing",
    "Content Marketing",
    "SEO",
    "SEM",
    "Google Ads",
    "Facebook Ads",
    "Social Media",
    "Branding",
    "PR",
    "Truyền thông",
    "Quảng cáo",
    "Content Creator",
    "Copywriting",
    "Email Marketing",
    "Influencer Marketing",
    "Google Analytics",
    "Marketing Automation",
    "Inbound Marketing",
    "Outbound Marketing",
    "TikTok",
    "YouTube",
    "Instagram",
    "LinkedIn",
    "Zalo",
    "Video Marketing",
    # ============ KẾ TOÁN / TÀI CHÍNH ============
    "Kế toán",
    "Kế toán tổng hợp",
    "Kế toán thuế",
    "Kế toán công nợ",
    "Kế toán kho",
    "Kiểm toán",
    "Tài chính",
    "Phân tích tài chính",
    "Báo cáo tài chính",
    "BCTC",
    "SAP",
    "MISA",
    "Fast Accounting",
    "Bravo",
    "ERP",
    "Thuế GTGT",
    "Thuế TNDN",
    "Excel nâng cao",
    "Pivot Table",
    "VLOOKUP",
    "Ngân sách",
    "Định giá",
    "Đầu tư",
    # ============ NHÂN SỰ ============
    "Nhân sự",
    "HR",
    "Tuyển dụng",
    "Recruitment",
    "Headhunter",
    "Đào tạo",
    "Training",
    "C&B",
    "Lương thưởng",
    "Phúc lợi",
    "BHXH",
    "BHYT",
    "Hợp đồng lao động",
    "Đánh giá nhân viên",
    "KPI",
    "OKR",
    "Văn hóa doanh nghiệp",
    "Employer Branding",
    # ============ HÀNH CHÍNH / VĂN PHÒNG ============
    "Hành chính",
    "Văn phòng",
    "Thư ký",
    "Trợ lý",
    "Lễ tân",
    "Admin",
    "Quản lý văn phòng",
    "Soạn thảo văn bản",
    "Lưu trữ hồ sơ",
    "Tiếp khách",
    # ============ NGÂN HÀNG / BẢO HIỂM ============
    "Ngân hàng",
    "Tín dụng",
    "Cho vay",
    "Thẩm định",
    "Bảo hiểm",
    "Bảo hiểm nhân thọ",
    "Bancassurance",
    "Tư vấn tài chính",
    "Đầu tư",
    "Chứng khoán",
    "Quỹ đầu tư",
    # ============ Y TẾ / DƯỢC ============
    "Y tế",
    "Dược",
    "Điều dưỡng",
    "Bác sĩ",
    "Y sĩ",
    "Dược sĩ",
    "Trình dược viên",
    "Chăm sóc sức khỏe",
    "Bệnh viện",
    "Phòng khám",
    "Thiết bị y tế",
    # ============ GIÁO DỤC / ĐÀO TẠO ============
    "Giáo dục",
    "Đào tạo",
    "Giảng dạy",
    "Giáo viên",
    "Gia sư",
    "IELTS",
    "TOEIC",
    "Tiếng Anh trẻ em",
    "Mầm non",
    "Tiểu học",
    "THCS",
    "THPT",
    "Đại học",
    # ============ KHÁCH SẠN / NHÀ HÀNG ============
    "Khách sạn",
    "Nhà hàng",
    "F&B",
    "Hospitality",
    "Lễ tân khách sạn",
    "Housekeeping",
    "Đầu bếp",
    "Phục vụ",
    "Bartender",
    "Barista",
    "Quản lý nhà hàng",
    # ============ XÂY DỰNG / BẤT ĐỘNG SẢN ============
    "Xây dựng",
    "Kiến trúc",
    "Thiết kế nội thất",
    "AutoCAD",
    "Revit",
    "SketchUp",
    "3D Max",
    "Giám sát công trình",
    "Kỹ sư xây dựng",
    "Dự toán",
    "Bất động sản",
    "Môi giới BĐS",
    "Tư vấn BĐS",
    "Định giá BĐS",
    # ============ CƠ KHÍ / ĐIỆN / SẢN XUẤT ============
    "Cơ khí",
    "Điện",
    "Điện tử",
    "Điện lạnh",
    "Tự động hóa",
    "PLC",
    "SCADA",
    "Bảo trì",
    "Bảo dưỡng",
    "Vận hành máy",
    "QA",
    "QC",
    "Quản lý chất lượng",
    "ISO",
    "5S",
    "Lean",
    "Six Sigma",
    "Kaizen",
    "Sản xuất",
    # ============ VẬN TẢI / LOGISTICS ============
    "Logistics",
    "Vận tải",
    "Xuất nhập khẩu",
    "XNK",
    "Hải quan",
    "Kho vận",
    "Supply Chain",
    "Chuỗi cung ứng",
    "Giao nhận",
    "Forwarder",
    "Lái xe",
    # ============ NGÔN NGỮ ============
    "Tiếng Anh",
    "Tiếng Nhật",
    "Tiếng Hàn",
    "Tiếng Trung",
    "Tiếng Pháp",
    "Tiếng Đức",
    "N1",
    "N2",
    "N3",
    "JLPT",
    "TOPIK",
    "HSK",
    "Biên phiên dịch",
    "Thông dịch",
    # ============ KỸ NĂNG MỀM / CHUNG ============
    "Tin học văn phòng",
    "MS Office",
    "Excel",
    "Word",
    "PowerPoint",
    "Google Sheets",
    "Kỹ năng giao tiếp",
    "Làm việc nhóm",
    "Teamwork",
    "Quản lý thời gian",
    "Giải quyết vấn đề",
    "Tư duy logic",
    "Chịu áp lực",
    "Đa nhiệm",
    "Thuyết trình",
    "Đàm phán",
    "Lãnh đạo",
    "Leadership",
    "Quản lý dự án",
    "PMP",
]

# ============================================
# DEFAULT VALUES
# ============================================
DEFAULT_WEBSITE = "https://ptit.edu.vn/"
DEFAULT_FOUNDED_YEAR = 2021
DEFAULT_PHONE = "0123456789"
DEFAULT_LOGO_URL = "https://firebasestorage.googleapis.com/v0/b/jobsconnect-dafde.firebasestorage.app/o/logos%2Fcmgweqlmw0002nyow0t3e8jgt%2F1761150525314.jpg?alt=media&token=0d37d58c-6edd-47c2-b64e-edc6d9f7f60a"


def normalize_company_name_to_email(name: str) -> str:
    """Convert company name to email format"""
    import unicodedata

    # Remove Vietnamese diacritics
    normalized = unicodedata.normalize("NFD", name)
    ascii_name = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    # Remove special characters, keep only alphanumeric and spaces
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", ascii_name)
    # Replace spaces with dots, lowercase
    email_prefix = clean.lower().replace(" ", ".")
    # Remove consecutive dots and trim
    email_prefix = re.sub(r"\.+", ".", email_prefix).strip(".")
    return f"{email_prefix}@gmail.com" if email_prefix else "company@gmail.com"


def parse_deadline(deadline_str: str) -> Optional[datetime]:
    """Parse deadline string to datetime"""
    if not deadline_str:
        return None
    try:
        # Try format: DD/MM/YYYY
        return datetime.strptime(deadline_str.strip(), "%d/%m/%Y")
    except ValueError:
        try:
            # Try format: YYYY-MM-DD
            return datetime.strptime(deadline_str.strip(), "%Y-%m-%d")
        except ValueError:
            return None


def parse_salary(salary_str: str) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse salary string to min, max amounts and negotiable flag"""
    if not salary_str or salary_str.lower() in ["thỏa thuận", "negotiable", ""]:
        return None, None, True

    # Remove commas and spaces
    salary_str = salary_str.replace(",", "").replace(" ", "")

    # Try to find numbers
    numbers = re.findall(r"\d+", salary_str)

    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1]), False
    elif len(numbers) == 1:
        return int(numbers[0]), int(numbers[0]), False

    return None, None, True


def map_enum(value: str, mapping: dict, default: Optional[str] = None) -> Optional[str]:
    """Map a string value to an enum using the provided mapping"""
    if not value:
        return default
    return mapping.get(value.lower().strip(), default)


def map_company_size(size_str: str) -> Optional[str]:
    """Map company size string to enum"""
    if not size_str:
        return None

    for key, value in COMPANY_SIZE_MAP.items():
        if key in size_str:
            return value

    return None


def split_text_to_items(text: str, delimiters: List[str] = None) -> List[str]:
    """Split text into items using various delimiters"""
    if not text or text.strip() == "":
        return []

    if delimiters is None:
        delimiters = ["\n", ";", ",", "•", "-", "●"]

    # Try each delimiter
    items = [text]
    for delim in delimiters:
        new_items = []
        for item in items:
            new_items.extend(item.split(delim))
        items = new_items

    # Clean and filter
    items = [item.strip() for item in items if item.strip() and len(item.strip()) > 2]
    return items


def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text by matching against common skills list"""
    if not text:
        return []

    found_skills = set()
    text_lower = text.lower()

    for skill in COMMON_SKILLS:
        # Case-insensitive search
        skill_lower = skill.lower()
        # Use word boundary matching to avoid partial matches
        # e.g., "R" shouldn't match "React"
        if len(skill) <= 2:
            # For short skills like "R", "C", "Go", require word boundaries
            pattern = r"\b" + re.escape(skill_lower) + r"\b"
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        else:
            if skill_lower in text_lower:
                found_skills.add(skill)

    return list(found_skills)


class DataImporter:
    def __init__(self):
        self.db = DatabaseConnection()
        self.company_cache = {}  # name -> id
        self.user_cache = {}  # (name, userid) -> id

    def get_or_create_company(
        self,
        cursor,
        name: str,
        description: str = None,
        size: str = None,
        address: str = None,
        industry: str = None,
    ) -> str:
        """Get existing company or create new one"""
        if name in self.company_cache:
            # If company exists in cache but we have new industry info, update it
            if industry:
                cursor.execute(
                    """
                    UPDATE companies SET industry = COALESCE(industry, %s), "updatedAt" = NOW()
                    WHERE name = %s AND industry IS NULL
                """,
                    (industry, name),
                )
            return self.company_cache[name]

        # Check if exists
        cursor.execute("SELECT id, industry FROM companies WHERE name = %s", (name,))
        result = cursor.fetchone()

        if result:
            self.company_cache[name] = result["id"]
            # Update industry if company doesn't have one yet
            if industry and not result.get("industry"):
                cursor.execute(
                    """
                    UPDATE companies SET industry = %s, "updatedAt" = NOW()
                    WHERE id = %s
                """,
                    (industry, result["id"]),
                )
            return result["id"]

        # Create new
        company_id = str(uuid.uuid4())
        company_size = map_company_size(size)
        company_email = normalize_company_name_to_email(name)

        cursor.execute(
            """
            INSERT INTO companies (
                id, name, description, "companySize", address, industry, status,
                website, "foundedYear", phone, email, "logoUrl",
                "createdAt", "updatedAt"
            )
            VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s, %s, %s, NOW(), NOW())
        """,
            (
                company_id,
                name,
                description,
                company_size,
                address,
                industry,
                DEFAULT_WEBSITE,
                DEFAULT_FOUNDED_YEAR,
                DEFAULT_PHONE,
                company_email,
                DEFAULT_LOGO_URL,
            ),
        )

        self.company_cache[name] = company_id
        logger.info(f"Created company: {name}")
        return company_id

    def get_or_create_user(
        self, cursor, name: str, userid: str, gender: str = None
    ) -> str:
        """Get existing user or create new one"""
        cache_key = (name, userid)
        if cache_key in self.user_cache:
            return self.user_cache[cache_key]

        # Generate email from userid
        email = f"user_{userid}@imported.local"

        # Check if exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        result = cursor.fetchone()

        if result:
            self.user_cache[cache_key] = result["id"]
            return result["id"]

        # Create new
        user_id = str(uuid.uuid4())
        gender_enum = map_enum(gender, GENDER_MAP)

        cursor.execute(
            """
            INSERT INTO users (id, email, "passwordHash", "fullName", gender, role, status, "createdAt", "updatedAt")
            VALUES (%s, %s, %s, %s, %s, 'CANDIDATE', 'ACTIVE', NOW(), NOW())
        """,
            (
                user_id,
                email,
                "$2b$10$4OTxGbe.sGx1OlQfJtfIaubeekNXfU7rMMe3kB4G5l927V2SA6CT6",
                name,
                gender_enum,
            ),
        )

        self.user_cache[cache_key] = user_id
        logger.info(f"Created user: {name}")
        return user_id

    def import_companies(self, csv_path: str, limit: int = None):
        """Import companies from CSV"""
        logger.info(f"Importing companies from {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if limit:
            rows = rows[:limit]

        imported = 0
        skipped = 0

        with self.db.get_cursor() as cursor:
            for row in rows:
                try:
                    self._import_company_row(cursor, row)
                    imported += 1
                except Exception as e:
                    logger.error(f"Error importing company: {e}")
                    skipped += 1

                if imported % 100 == 0 and imported > 0:
                    logger.info(f"Imported {imported} companies...")

        logger.info(
            f"Companies import complete: {imported} imported, {skipped} skipped"
        )

    def _import_company_row(self, cursor, row: dict):
        """Import a single company row"""
        company_name = row.get("Name Company", "").strip()
        if not company_name:
            return

        # Use get_or_create to avoid duplicates
        self.get_or_create_company(
            cursor,
            name=company_name,
            description=row.get("Company Overview"),
            size=row.get("Company Size"),
            address=row.get("Company Address"),
        )

    def import_jobs(self, csv_path: str, limit: int = None):
        """Import jobs from CSV"""
        logger.info(f"Importing jobs from {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if limit:
            rows = rows[:limit]

        imported = 0
        skipped = 0

        with self.db.get_cursor() as cursor:
            for row in rows:
                try:
                    self._import_job_row(cursor, row)
                    imported += 1
                except Exception as e:
                    logger.error(f"Error importing job: {e}")
                    skipped += 1

                if imported % 100 == 0:
                    logger.info(f"Imported {imported} jobs...")

        logger.info(f"Jobs import complete: {imported} imported, {skipped} skipped")

    def _import_job_row(self, cursor, row: dict):
        """Import a single job row"""
        job_id = str(uuid.uuid4())

        # Get or create company
        company_name = row.get("Name Company", "Unknown Company")
        company_id = self.get_or_create_company(
            cursor,
            name=company_name,
            description=row.get("Company Overview"),
            size=row.get("Company Size"),
            address=row.get("Company Address"),
            industry=row.get("Industry"),
        )

        # Parse salary
        min_salary, max_salary, is_negotiable = parse_salary(row.get("Salary", ""))

        # Map enums
        experience_level = map_enum(
            row.get("Career Level") or row.get("Years of Experience"),
            EXPERIENCE_LEVEL_MAP,
        )
        job_type = map_enum(row.get("Job Type"), JOB_TYPE_MAP, "FULL_TIME")

        # Get job title and description with defaults
        job_title = row.get("Job Title", "").strip() or "Untitled"
        job_description = (
            row.get("Job Description", "").strip() or f"Mô tả công việc: {job_title}"
        )

        # Parse deadline
        expires_at = parse_deadline(row.get("Submission Deadline", ""))

        # Insert job (embedding sẽ được tạo sau)
        cursor.execute(
            """
            INSERT INTO jobs (
                id, "companyId", title, description, location, industry,
                "experienceLevel", type, status, "expiresAt", "createdAt", "updatedAt"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'ACTIVE', %s, NOW(), NOW())
        """,
            (
                job_id,
                company_id,
                job_title,
                job_description,
                row.get("Job Address"),
                row.get("Industry"),
                experience_level,
                job_type,
                expires_at,
            ),
        )

        # Insert salary
        if min_salary or max_salary or is_negotiable:
            cursor.execute(
                """
                INSERT INTO salaries (id, "jobId", "minAmount", "maxAmount", "isNegotiable", "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            """,
                (str(uuid.uuid4()), job_id, min_salary, max_salary, is_negotiable),
            )

        # Insert requirements
        requirements = split_text_to_items(row.get("Job Requirements", ""))
        for req in requirements[:10]:  # Limit to 10
            cursor.execute(
                """
                INSERT INTO job_requirements (id, "jobId", title, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, NOW(), NOW())
            """,
                (str(uuid.uuid4()), job_id, req[:500]),
            )

        # Insert benefits
        benefits = split_text_to_items(row.get("Benefits", ""))
        for benefit in benefits[:10]:  # Limit to 10
            cursor.execute(
                """
                INSERT INTO job_benefits (id, "jobId", title, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, NOW(), NOW())
            """,
                (str(uuid.uuid4()), job_id, benefit[:500]),
            )

        # Extract and insert skills from both Job Requirements and Job Description
        combined_text = (
            f"{row.get('Job Requirements', '')} {row.get('Job Description', '')}"
        )
        skills = extract_skills_from_text(combined_text)
        for skill in skills[:20]:  # Limit to 20 skills per job
            cursor.execute(
                """
                INSERT INTO job_skills (id, "jobId", "skillName", "createdAt", "updatedAt")
                VALUES (%s, %s, %s, NOW(), NOW())
            """,
                (str(uuid.uuid4()), job_id, skill[:100]),
            )

    def import_cvs(self, csv_path: str, limit: int = None, random_sample: bool = True):
        """Import CVs from CSV"""
        logger.info(f"Importing CVs from {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Default limit for CVs is 5000 (random sample)
        if limit is None:
            limit = 5000

        if limit and len(rows) > limit:
            if random_sample:
                logger.info(f"Randomly sampling {limit} CVs from {len(rows)} total")
                rows = random.sample(rows, limit)
            else:
                rows = rows[:limit]

        imported = 0
        skipped = 0

        with self.db.get_cursor() as cursor:
            for row in rows:
                try:
                    self._import_cv_row(cursor, row)
                    imported += 1
                except Exception as e:
                    logger.error(f"Error importing CV: {e}")
                    skipped += 1

                if imported % 100 == 0:
                    logger.info(f"Imported {imported} CVs...")

        logger.info(f"CVs import complete: {imported} imported, {skipped} skipped")

    def _import_cv_row(self, cursor, row: dict):
        """Import a single CV row"""
        cv_id = str(uuid.uuid4())

        # Get or create user
        user_name = row.get("user_name", "Unknown")
        userid = row.get("userid", str(uuid.uuid4()))
        gender = row.get("gender")

        user_id = self.get_or_create_user(cursor, user_name, userid, gender)

        # CV title from desired_job_translated
        cv_title = row.get("desired_job_translated", "CV của " + user_name)

        # Build summary from available info
        summary_parts = []
        if row.get("industry"):
            summary_parts.append(f"Ngành: {row['industry']}")
        if row.get("workplace_desired"):
            summary_parts.append(f"Nơi làm việc: {row['workplace_desired']}")
        if row.get("desired_salary"):
            summary_parts.append(f"Mức lương mong muốn: {row['desired_salary']}")
        summary = ". ".join(summary_parts) if summary_parts else None

        # Map experience level
        work_exp = row.get("work_experience", "")
        experience_text = map_enum(work_exp, EXPERIENCE_LEVEL_MAP)

        # Insert CV
        cursor.execute(
            """
            INSERT INTO cvs (
                id, "userId", title, "isMain", "fullName", "currentPosition",
                summary, "isOpenForJob", "createdAt", "updatedAt"
            ) VALUES (%s, %s, %s, true, %s, %s, %s, true, NOW(), NOW())
        """,
            (cv_id, user_id, cv_title[:255], user_name, cv_title[:255], summary),
        )

        # Insert skills
        skills_text = row.get("Skills", "")
        skills = split_text_to_items(skills_text)
        for skill in skills[:20]:  # Limit to 20
            cursor.execute(
                """
                INSERT INTO cv_skills (id, "cvId", "skillName", level, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, 'INTERMEDIATE', NOW(), NOW())
            """,
                (str(uuid.uuid4()), cv_id, skill[:100]),
            )

        # Insert work experience
        exp_text = row.get("Experience", "")
        experiences = split_text_to_items(exp_text)

        # Also add work_experience as a record
        if work_exp and work_exp.strip():
            experiences.insert(0, work_exp)

        for exp in experiences[:10]:  # Limit to 10
            cursor.execute(
                """
                INSERT INTO work_experiences (id, "cvId", title, company, "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, NOW(), NOW())
            """,
                (str(uuid.uuid4()), cv_id, exp[:255], ""),
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Import CSV data to database")
    parser.add_argument("--companies", type=str, help="Path to companies CSV file")
    parser.add_argument("--jobs", type=str, help="Path to jobs CSV file")
    parser.add_argument("--cvs", type=str, help="Path to CVs CSV file")
    parser.add_argument("--limit", type=int, help="Limit number of records to import")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Import all default files (companies -> jobs -> cvs)",
    )

    args = parser.parse_args()

    importer = DataImporter()

    # Test connection
    if not importer.db.test_connection():
        logger.error("Cannot connect to database")
        sys.exit(1)

    data_dir = Path(__file__).parent.parent / "data"

    if args.all:
        # Import in order: companies → jobs → cvs
        companies_path = data_dir / "company_data.csv"
        jobs_path = data_dir / "job_data.csv"
        cvs_path = data_dir / "candidates_dataset.csv"

        if companies_path.exists():
            importer.import_companies(str(companies_path), args.limit)
        if jobs_path.exists():
            importer.import_jobs(str(jobs_path), args.limit)
        if cvs_path.exists():
            importer.import_cvs(str(cvs_path), args.limit)
    else:
        if args.companies:
            importer.import_companies(args.companies, args.limit)
        if args.jobs:
            importer.import_jobs(args.jobs, args.limit)
        if args.cvs:
            importer.import_cvs(args.cvs, args.limit)

    if not args.companies and not args.jobs and not args.cvs and not args.all:
        parser.print_help()


if __name__ == "__main__":
    main()
