"""
Visualize a FEW jobs and CVs with CLEAR LABELS to show how similar items cluster together.
This creates easy-to-understand visualizations with job/CV titles displayed.

Usage:
    python scripts/visualize_with_labels.py
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import rcParams

from recommend_service.database.connection import DatabaseConnection

# Configure for Vietnamese text
rcParams['font.family'] = 'DejaVu Sans'

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def truncate_title(title: str, max_len: int = 30) -> str:
    """Truncate title for display"""
    if not title:
        return "No title"
    if len(title) <= max_len:
        return title
    return title[:max_len-3] + "..."


def load_jobs_by_industry(db: DatabaseConnection, jobs_per_industry: int = 30, top_industries: int = 10):
    """Load jobs grouped by their ACTUAL industry field from database"""

    # First, get the top industries by job count
    industry_query = """
        SELECT industry, COUNT(*) as cnt
        FROM jobs
        WHERE industry IS NOT NULL AND industry != '' AND "titleEmbedding" IS NOT NULL
        GROUP BY industry
        ORDER BY cnt DESC
        LIMIT %s
    """

    with db.get_cursor() as cursor:
        cursor.execute(industry_query, (top_industries,))
        industries = cursor.fetchall()

    jobs_by_industry = {}

    for ind in industries:
        industry_name = ind["industry"]

        # Get random jobs from this industry
        query = """
            SELECT id, title, industry, "titleEmbedding"
            FROM (
                SELECT id, title, industry, "titleEmbedding"
                FROM jobs
                WHERE industry = %s AND "titleEmbedding" IS NOT NULL
            ) sub
            ORDER BY RANDOM()
            LIMIT %s
        """

        with db.get_cursor() as cursor:
            cursor.execute(query, (industry_name, jobs_per_industry))
            rows = cursor.fetchall()

        jobs = []
        for row in rows:
            emb = row["titleEmbedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            if emb:
                jobs.append({
                    "id": row["id"],
                    "title": row["title"],
                    "industry": industry_name,
                    "embedding": np.array(emb)
                })

        if jobs:
            jobs_by_industry[industry_name] = jobs
            logger.info(f"  {industry_name}: {len(jobs)} jobs")

    return jobs_by_industry


def visualize_jobs_by_industry(jobs_by_industry: dict, output_path: str, labels_per_industry: int = 2):
    """Create visualization showing job clusters by INDUSTRY (from database)"""

    # Flatten jobs
    all_jobs = []
    for industry, jobs in jobs_by_industry.items():
        all_jobs.extend(jobs)

    if len(all_jobs) < 5:
        logger.warning("Not enough jobs for visualization")
        return

    # Get embeddings and reduce to 2D
    embeddings = np.array([j["embedding"] for j in all_jobs])

    perplexity = min(30, max(5, len(all_jobs) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(22, 18))

    # Use distinct, saturated colors for industries
    distinct_colors = [
        '#e41a1c',  # Red
        '#377eb8',  # Blue
        '#4daf4a',  # Green
        '#984ea3',  # Purple
        '#ff7f00',  # Orange
        '#ffff33',  # Yellow
        '#a65628',  # Brown
        '#f781bf',  # Pink
        '#00ced1',  # Dark Cyan
        '#8b0000',  # Dark Red
        '#006400',  # Dark Green
        '#00008b',  # Dark Blue
        '#ff1493',  # Deep Pink
        '#ffd700',  # Gold
        '#32cd32',  # Lime Green
    ]
    industry_colors = {ind: distinct_colors[i % len(distinct_colors)] for i, ind in enumerate(jobs_by_industry.keys())}

    # Plot each industry
    idx = 0
    for industry, jobs in jobs_by_industry.items():
        ind_coords = coords[idx:idx+len(jobs)]
        color = industry_colors[industry]

        # Plot all points
        ax.scatter(ind_coords[:, 0], ind_coords[:, 1],
                  c=[color], s=80, alpha=0.85,
                  label=f"{truncate_title(industry, 25)} ({len(jobs)})",
                  edgecolors='black', linewidth=0.5)

        # Add labels for some jobs
        n_labels = min(labels_per_industry, len(jobs))
        label_indices = np.linspace(0, len(jobs)-1, n_labels, dtype=int)

        for i in label_indices:
            job = jobs[i]
            title = truncate_title(job["title"], 35)
            ax.annotate(title,
                       (ind_coords[i, 0], ind_coords[i, 1]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=8, alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

        idx += len(jobs)

    ax.set_title(f"Job Clustering by INDUSTRY - {len(all_jobs)} Jobs\n(Same industry should cluster together)",
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def load_sample_jobs_by_category(db: DatabaseConnection, jobs_per_category: int = 22):
    """Load jobs from different categories for clear visualization"""

    # Define keyword patterns for different job types
    categories = {
        "Backend Developer": ["backend", "python developer", "java developer", "nodejs", "php developer", ".net", "golang"],
        "Frontend Developer": ["frontend", "react", "angular", "vue", "javascript", "typescript"],
        "Data/AI": ["data engineer", "data scientist", "machine learning", "ai", "data analyst", "big data"],
        "Marketing": ["marketing", "seo", "content", "digital marketing", "brand"],
        "Sales": ["sales", "business development", "account manager", "kinh doanh"],
        "HR/Admin": ["hr", "human resource", "admin", "tuyển dụng", "nhân sự"],
        "Design": ["design", "ui", "ux", "graphic", "creative"],
        "QA/Tester": ["tester", "qa", "quality", "test engineer"],
    }

    jobs_by_cat = {}

    for cat_name, keywords in categories.items():
        # Build query with keywords
        keyword_conditions = " OR ".join([f"LOWER(title) LIKE '%{kw}%'" for kw in keywords])
        query = f"""
            SELECT id, title, "titleEmbedding"
            FROM jobs
            WHERE "titleEmbedding" IS NOT NULL
            AND ({keyword_conditions})
            LIMIT {jobs_per_category}
        """

        with db.get_cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

        jobs = []
        for row in rows:
            emb = row["titleEmbedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            if emb:
                jobs.append({
                    "id": row["id"],
                    "title": row["title"],
                    "category": cat_name,
                    "embedding": np.array(emb)
                })

        if jobs:
            jobs_by_cat[cat_name] = jobs
            logger.info(f"  {cat_name}: {len(jobs)} jobs")

    return jobs_by_cat


def load_sample_cvs_with_recommendations(db: DatabaseConnection, job_ids: list, limit: int = 10):
    """Load CVs that have recommendations to the sampled jobs"""

    if not job_ids:
        return [], []

    # Get CVs that have recommendations
    query = """
        SELECT DISTINCT c.id, c.title, c."titleEmbedding", r."jobId", r.similarity
        FROM cvs c
        INNER JOIN recommend_jobs_for_cv r ON c.id = r."cvId"
        WHERE c."titleEmbedding" IS NOT NULL
        AND c."isMain" = true
        AND r."jobId" = ANY(%s)
        ORDER BY r.similarity DESC
        LIMIT %s
    """

    with db.get_cursor() as cursor:
        cursor.execute(query, (job_ids, limit * 3))
        rows = cursor.fetchall()

    cvs = {}
    recommendations = []

    for row in rows:
        cv_id = row["id"]
        if cv_id not in cvs:
            emb = row["titleEmbedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            if emb:
                cvs[cv_id] = {
                    "id": cv_id,
                    "title": row["title"],
                    "embedding": np.array(emb)
                }

        recommendations.append({
            "cvId": cv_id,
            "jobId": row["jobId"],
            "similarity": float(row["similarity"])
        })

    return list(cvs.values())[:limit], recommendations


def load_similar_job_pairs(db: DatabaseConnection, job_ids: list):
    """Load similar job pairs for the sampled jobs"""

    if not job_ids:
        return []

    query = """
        SELECT "jobId", "similarJobId", similarity
        FROM similar_jobs
        WHERE "jobId" = ANY(%s) AND "similarJobId" = ANY(%s)
        ORDER BY similarity DESC
    """

    with db.get_cursor() as cursor:
        cursor.execute(query, (job_ids, job_ids))
        return cursor.fetchall()


def visualize_jobs_with_labels(jobs_by_cat: dict, output_path: str, labels_per_cat: int = 3):
    """Create visualization showing job clusters with titles (selective labels)"""

    # Flatten jobs
    all_jobs = []
    for cat, jobs in jobs_by_cat.items():
        all_jobs.extend(jobs)

    if len(all_jobs) < 5:
        logger.warning("Not enough jobs for visualization")
        return

    # Get embeddings and reduce to 2D
    embeddings = np.array([j["embedding"] for j in all_jobs])

    perplexity = min(30, max(5, len(all_jobs) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))

    # Colors for categories
    colors = {
        "Backend Developer": "#2ecc71",
        "Frontend Developer": "#3498db",
        "Data/AI": "#9b59b6",
        "Marketing": "#e74c3c",
        "Sales": "#f39c12",
        "HR/Admin": "#1abc9c",
        "Design": "#e91e63",
        "QA/Tester": "#00bcd4",
    }

    # Plot each category
    idx = 0
    for cat, jobs in jobs_by_cat.items():
        cat_coords = coords[idx:idx+len(jobs)]
        color = colors.get(cat, "#95a5a6")

        # Plot all points
        ax.scatter(cat_coords[:, 0], cat_coords[:, 1],
                  c=color, s=120, alpha=0.7,
                  label=f"{cat} ({len(jobs)})", edgecolors='white', linewidth=1.5)

        # Add labels for only some jobs (spread across the cluster)
        # Pick jobs at different positions to show variety
        n_labels = min(labels_per_cat, len(jobs))
        label_indices = np.linspace(0, len(jobs)-1, n_labels, dtype=int)

        for i in label_indices:
            job = jobs[i]
            title = truncate_title(job["title"], 40)
            ax.annotate(title,
                       (cat_coords[i, 0], cat_coords[i, 1]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, alpha=0.95,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

        idx += len(jobs)

    ax.set_title(f"Job Clustering by Category - {len(all_jobs)} Jobs\n(Similar jobs should cluster together)",
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def visualize_similar_jobs_with_labels(jobs_by_cat: dict, similar_pairs: list, output_path: str):
    """Visualize similar jobs with lines connecting them and labels"""

    # Flatten jobs
    all_jobs = []
    for cat, jobs in jobs_by_cat.items():
        all_jobs.extend(jobs)

    if len(all_jobs) < 5:
        return

    # Create job ID mapping
    job_id_to_idx = {j["id"]: i for i, j in enumerate(all_jobs)}
    job_id_to_job = {j["id"]: j for j in all_jobs}

    # Get embeddings and reduce to 2D
    embeddings = np.array([j["embedding"] for j in all_jobs])
    perplexity = min(15, max(5, len(all_jobs) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))

    colors = {
        "Backend Developer": "#2ecc71",
        "Frontend Developer": "#3498db",
        "Data/AI": "#9b59b6",
        "Marketing": "#e74c3c",
        "Sales": "#f39c12",
        "HR/Admin": "#1abc9c",
        "Design": "#e91e63",
        "QA/Tester": "#00bcd4",
    }

    # Draw similarity lines first (top 3 per job)
    from collections import defaultdict
    job_similar = defaultdict(list)

    for pair in similar_pairs:
        job_id = pair["jobId"]
        similar_id = pair["similarJobId"]
        if job_id in job_id_to_idx and similar_id in job_id_to_idx:
            job_similar[job_id].append((similar_id, pair["similarity"]))

    lines_info = []
    for job_id, similars in job_similar.items():
        similars.sort(key=lambda x: x[1], reverse=True)
        for similar_id, sim in similars[:2]:  # Top 2 per job
            idx1 = job_id_to_idx[job_id]
            idx2 = job_id_to_idx[similar_id]

            ax.plot([coords[idx1, 0], coords[idx2, 0]],
                   [coords[idx1, 1], coords[idx2, 1]],
                   '-', color='#27ae60', alpha=0.6, linewidth=2)

            # Add similarity score at midpoint
            mid_x = (coords[idx1, 0] + coords[idx2, 0]) / 2
            mid_y = (coords[idx1, 1] + coords[idx2, 1]) / 2
            ax.annotate(f"{sim:.2f}", (mid_x, mid_y), fontsize=7,
                       color='green', fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            lines_info.append((job_id_to_job[job_id]["title"],
                             job_id_to_job[similar_id]["title"], sim))

    # Plot points with labels
    idx = 0
    for cat, jobs in jobs_by_cat.items():
        cat_coords = coords[idx:idx+len(jobs)]
        color = colors.get(cat, "#95a5a6")

        ax.scatter(cat_coords[:, 0], cat_coords[:, 1],
                  c=color, s=250, alpha=0.8,
                  label=cat, edgecolors='white', linewidth=2,
                  zorder=10)

        for i, job in enumerate(jobs):
            title = truncate_title(job["title"], 30)
            ax.annotate(title,
                       (cat_coords[i, 0], cat_coords[i, 1]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=8, alpha=0.95,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.4),
                       zorder=11)

        idx += len(jobs)

    ax.set_title("Similar Jobs Network\n(Green lines connect similar jobs, numbers show similarity score)",
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")

    # Print some similar pairs
    if lines_info:
        logger.info("\nTop similar job pairs visualized:")
        for j1, j2, sim in sorted(lines_info, key=lambda x: x[2], reverse=True)[:10]:
            logger.info(f"  [{sim:.3f}] '{truncate_title(j1, 40)}' <-> '{truncate_title(j2, 40)}'")


def load_cvs_with_their_recommended_jobs(db: DatabaseConnection, num_cvs: int = 5, jobs_per_cv: int = 5):
    """Load CVs and ONLY the jobs that are recommended for them"""

    # Get RANDOM CVs that have recommendations (different each run)
    cv_query = """
        SELECT id, title, "titleEmbedding"
        FROM (
            SELECT DISTINCT c.id, c.title, c."titleEmbedding"
            FROM cvs c
            INNER JOIN recommend_jobs_for_cv r ON c.id = r."cvId"
            WHERE c."titleEmbedding" IS NOT NULL AND c."isMain" = true
        ) sub
        ORDER BY RANDOM()
        LIMIT %s
    """

    with db.get_cursor() as cursor:
        cursor.execute(cv_query, (num_cvs,))
        cv_rows = cursor.fetchall()

    cvs = []
    for row in cv_rows:
        emb = row["titleEmbedding"]
        if isinstance(emb, str):
            emb = json.loads(emb)
        if emb:
            cvs.append({
                "id": row["id"],
                "title": row["title"],
                "embedding": np.array(emb)
            })

    if not cvs:
        return [], [], []

    cv_ids = [c["id"] for c in cvs]

    # Get recommendations for these CVs (top jobs_per_cv per CV)
    rec_query = """
        SELECT r."cvId", r."jobId", r.similarity, j.title, j."titleEmbedding"
        FROM recommend_jobs_for_cv r
        INNER JOIN jobs j ON r."jobId" = j.id
        WHERE r."cvId" = ANY(%s)
        AND j."titleEmbedding" IS NOT NULL
        ORDER BY r."cvId", r.similarity DESC
    """

    with db.get_cursor() as cursor:
        cursor.execute(rec_query, (cv_ids,))
        rec_rows = cursor.fetchall()

    # Group by CV and take top jobs_per_cv
    from collections import defaultdict
    cv_recs = defaultdict(list)
    for row in rec_rows:
        cv_id = row["cvId"]
        if len(cv_recs[cv_id]) < jobs_per_cv:
            cv_recs[cv_id].append(row)

    # Build jobs list and recommendations
    jobs = {}
    recommendations = []

    for cv_id, recs in cv_recs.items():
        for row in recs:
            job_id = row["jobId"]
            if job_id not in jobs:
                emb = row["titleEmbedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                if emb:
                    jobs[job_id] = {
                        "id": job_id,
                        "title": row["title"],
                        "embedding": np.array(emb),
                        "cv_id": cv_id  # Track which CV this job is recommended for
                    }

            recommendations.append({
                "cvId": cv_id,
                "jobId": job_id,
                "similarity": float(row["similarity"])
            })

    return cvs, list(jobs.values()), recommendations


def visualize_cv_job_recommendations_clean(cvs: list, jobs: list, recommendations: list, output_path: str):
    """Clean visualization: CVs and ONLY their recommended jobs"""

    if not cvs or not jobs:
        logger.warning("No CVs or jobs to visualize")
        return

    # Combine embeddings
    cv_embeddings = np.array([c["embedding"] for c in cvs])
    job_embeddings = np.array([j["embedding"] for j in jobs])
    all_embeddings = np.vstack([cv_embeddings, job_embeddings])

    # Reduce dimensions
    perplexity = min(15, max(5, len(all_embeddings) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(all_embeddings)

    cv_coords = coords[:len(cvs)]
    job_coords = coords[len(cvs):]

    # Create mappings
    cv_id_to_idx = {c["id"]: i for i, c in enumerate(cvs)}
    job_id_to_idx = {j["id"]: i for i, j in enumerate(jobs)}
    cv_id_to_cv = {c["id"]: c for c in cvs}

    # Assign colors to CVs
    cv_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
                 '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b']

    cv_color_map = {cv["id"]: cv_colors[i % len(cv_colors)] for i, cv in enumerate(cvs)}

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw recommendation lines
    for rec in recommendations:
        cv_id = rec["cvId"]
        job_id = rec["jobId"]
        if cv_id in cv_id_to_idx and job_id in job_id_to_idx:
            cv_idx = cv_id_to_idx[cv_id]
            job_idx = job_id_to_idx[job_id]
            sim = rec["similarity"]
            color = cv_color_map[cv_id]

            ax.plot([cv_coords[cv_idx, 0], job_coords[job_idx, 0]],
                   [cv_coords[cv_idx, 1], job_coords[job_idx, 1]],
                   '-', color=color, alpha=0.6, linewidth=2)

            # Add similarity score near job
            mid_x = cv_coords[cv_idx, 0] * 0.3 + job_coords[job_idx, 0] * 0.7
            mid_y = cv_coords[cv_idx, 1] * 0.3 + job_coords[job_idx, 1] * 0.7
            ax.annotate(f"{sim:.2f}", (mid_x, mid_y), fontsize=7,
                       color=color, fontweight='bold', alpha=0.8)

    # Plot jobs with labels
    for i, job in enumerate(jobs):
        # Find which CV this job is recommended for (use first one)
        job_cv_id = job.get("cv_id")
        color = cv_color_map.get(job_cv_id, '#95a5a6')

        ax.scatter(job_coords[i, 0], job_coords[i, 1],
                  c=color, s=150, alpha=0.7,
                  edgecolors='white', linewidth=1.5, marker='o', zorder=5)

        title = truncate_title(job["title"], 30)
        ax.annotate(title,
                   (job_coords[i, 0], job_coords[i, 1]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=8, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3),
                   zorder=6)

    # Plot CVs (larger, star markers)
    for i, cv in enumerate(cvs):
        color = cv_color_map[cv["id"]]
        ax.scatter(cv_coords[i, 0], cv_coords[i, 1],
                  c=color, s=400, alpha=1.0,
                  edgecolors='black', linewidth=2,
                  marker='*', zorder=10)

        title = truncate_title(cv["title"], 25)
        ax.annotate(f"CV: {title}",
                   (cv_coords[i, 0], cv_coords[i, 1]),
                   xytext=(12, 12), textcoords='offset points',
                   fontsize=10, alpha=1.0, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.8, edgecolor='black'),
                   zorder=11)

    ax.set_title(f"CV-Job Recommendations\n({len(cvs)} CVs, {len(jobs)} recommended jobs, lines show similarity scores)",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)

    # Add legend for CVs
    legend_elements = []
    for cv in cvs:
        color = cv_color_map[cv["id"]]
        title = truncate_title(cv["title"], 20)
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=color,
                                      markersize=15, label=f"CV: {title}"))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def load_seed_jobs_and_their_similars(db: DatabaseConnection, num_seeds: int = 10, similars_per_seed: int = 10):
    """Load seed jobs and ALL their similar jobs from DB"""

    # Get RANDOM seed jobs that have similar jobs (different categories each run)
    seed_query = """
        SELECT id, title, "titleEmbedding"
        FROM (
            SELECT DISTINCT j.id, j.title, j."titleEmbedding"
            FROM jobs j
            INNER JOIN similar_jobs s ON j.id = s."jobId"
            WHERE j."titleEmbedding" IS NOT NULL
        ) sub
        ORDER BY RANDOM()
        LIMIT %s
    """

    with db.get_cursor() as cursor:
        cursor.execute(seed_query, (num_seeds,))
        seed_rows = cursor.fetchall()

    seed_jobs = []
    for row in seed_rows:
        emb = row["titleEmbedding"]
        if isinstance(emb, str):
            emb = json.loads(emb)
        if emb:
            seed_jobs.append({
                "id": row["id"],
                "title": row["title"],
                "embedding": np.array(emb),
                "is_seed": True
            })

    if not seed_jobs:
        return [], [], []

    seed_ids = [j["id"] for j in seed_jobs]

    # Get similar jobs for these seeds
    similar_query = """
        SELECT s."jobId", s."similarJobId", s.similarity, j.title, j."titleEmbedding"
        FROM similar_jobs s
        INNER JOIN jobs j ON s."similarJobId" = j.id
        WHERE s."jobId" = ANY(%s)
        AND j."titleEmbedding" IS NOT NULL
        ORDER BY s."jobId", s.similarity DESC
    """

    with db.get_cursor() as cursor:
        cursor.execute(similar_query, (seed_ids,))
        similar_rows = cursor.fetchall()

    # Group by seed and take top similars_per_seed
    from collections import defaultdict
    seed_similars = defaultdict(list)
    for row in similar_rows:
        seed_id = row["jobId"]
        if len(seed_similars[seed_id]) < similars_per_seed:
            seed_similars[seed_id].append(row)

    # Build similar jobs list
    similar_jobs = {}
    similarities = []

    for seed_id, similars in seed_similars.items():
        for row in similars:
            similar_id = row["similarJobId"]
            if similar_id not in similar_jobs and similar_id not in seed_ids:
                emb = row["titleEmbedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                if emb:
                    similar_jobs[similar_id] = {
                        "id": similar_id,
                        "title": row["title"],
                        "embedding": np.array(emb),
                        "is_seed": False,
                        "seed_id": seed_id  # Track which seed this is similar to
                    }

            similarities.append({
                "seedId": seed_id,
                "similarId": similar_id,
                "similarity": float(row["similarity"])
            })

    return seed_jobs, list(similar_jobs.values()), similarities


def visualize_seed_and_similar_jobs(seed_jobs: list, similar_jobs: list, similarities: list, output_path: str):
    """Visualize seed jobs and their similar jobs - seeds should be surrounded by their similars"""

    if not seed_jobs or not similar_jobs:
        logger.warning("No seed jobs or similar jobs to visualize")
        return

    # Combine all jobs
    all_jobs = seed_jobs + similar_jobs

    # Get embeddings
    embeddings = np.array([j["embedding"] for j in all_jobs])

    # Reduce dimensions
    perplexity = min(30, max(5, len(all_jobs) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    seed_coords = coords[:len(seed_jobs)]
    similar_coords = coords[len(seed_jobs):]

    # Create mappings
    seed_id_to_idx = {j["id"]: i for i, j in enumerate(seed_jobs)}
    similar_id_to_idx = {j["id"]: i for i, j in enumerate(similar_jobs)}

    # Assign colors to seeds
    seed_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
                   '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#795548']

    seed_color_map = {job["id"]: seed_colors[i % len(seed_colors)] for i, job in enumerate(seed_jobs)}

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14))

    # Draw similarity lines first
    for sim in similarities:
        seed_id = sim["seedId"]
        similar_id = sim["similarId"]
        if seed_id in seed_id_to_idx and similar_id in similar_id_to_idx:
            seed_idx = seed_id_to_idx[seed_id]
            similar_idx = similar_id_to_idx[similar_id]
            score = sim["similarity"]
            color = seed_color_map[seed_id]

            ax.plot([seed_coords[seed_idx, 0], similar_coords[similar_idx, 0]],
                   [seed_coords[seed_idx, 1], similar_coords[similar_idx, 1]],
                   '-', color=color, alpha=0.4, linewidth=1.5)

    # Plot similar jobs (smaller circles)
    for i, job in enumerate(similar_jobs):
        seed_id = job.get("seed_id")
        color = seed_color_map.get(seed_id, '#95a5a6')

        ax.scatter(similar_coords[i, 0], similar_coords[i, 1],
                  c=color, s=100, alpha=0.6,
                  edgecolors='white', linewidth=1, marker='o', zorder=5)

        title = truncate_title(job["title"], 25)
        ax.annotate(title,
                   (similar_coords[i, 0], similar_coords[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=7, alpha=0.85,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.25),
                   zorder=6)

    # Plot seed jobs (larger stars)
    for i, job in enumerate(seed_jobs):
        color = seed_color_map[job["id"]]
        ax.scatter(seed_coords[i, 0], seed_coords[i, 1],
                  c=color, s=500, alpha=1.0,
                  edgecolors='black', linewidth=2,
                  marker='*', zorder=10)

        title = truncate_title(job["title"], 30)
        ax.annotate(f"SEED: {title}",
                   (seed_coords[i, 0], seed_coords[i, 1]),
                   xytext=(12, 12), textcoords='offset points',
                   fontsize=9, alpha=1.0, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9, edgecolor='black'),
                   zorder=11)

    ax.set_title(f"Similar Jobs Clustering\n({len(seed_jobs)} seed jobs ★, {len(similar_jobs)} similar jobs ●)\nSimilar jobs should cluster around their seed job",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for job in seed_jobs:
        color = seed_color_map[job["id"]]
        title = truncate_title(job["title"], 20)
        legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor=color,
                                      markersize=15, label=f"Seed: {title}"))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")

    # Print stats
    logger.info(f"\nSeed jobs and their similar jobs:")
    seed_id_to_job = {j["id"]: j for j in seed_jobs}
    from collections import defaultdict
    seed_counts = defaultdict(int)
    for sim in similarities:
        seed_counts[sim["seedId"]] += 1

    for seed_id, count in seed_counts.items():
        seed_title = seed_id_to_job[seed_id]["title"]
        logger.info(f"  ★ '{truncate_title(seed_title, 40)}' -> {count} similar jobs")


def visualize_industry_similarity_heatmap(jobs_by_industry: dict, output_path: str):
    """Create heatmap showing average similarity between industries"""
    import seaborn as sns

    industries = list(jobs_by_industry.keys())
    n = len(industries)

    # Calculate average embedding for each industry
    industry_centroids = {}
    for ind, jobs in jobs_by_industry.items():
        embeddings = np.array([j["embedding"] for j in jobs])
        industry_centroids[ind] = embeddings.mean(axis=0)

    # Calculate cosine similarity between all pairs
    similarity_matrix = np.zeros((n, n))
    for i, ind1 in enumerate(industries):
        for j, ind2 in enumerate(industries):
            c1 = industry_centroids[ind1]
            c2 = industry_centroids[ind2]
            sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
            similarity_matrix[i, j] = sim

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    # Truncate industry names for display
    short_names = [truncate_title(ind, 20) for ind in industries]

    sns.heatmap(similarity_matrix,
                xticklabels=short_names,
                yticklabels=short_names,
                annot=True, fmt='.2f',
                cmap='RdYlGn',
                vmin=0.5, vmax=1.0,
                ax=ax,
                annot_kws={'size': 9})

    ax.set_title("Industry Similarity Heatmap\n(Higher = More Similar Embeddings)",
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def visualize_recommendation_quality(db: DatabaseConnection, output_path: str, num_cvs: int = 10):
    """Show that recommended jobs have HIGHER similarity than random jobs"""

    # Get CVs with recommendations
    cv_query = """
        SELECT c.id, c.title, c."titleEmbedding"
        FROM (
            SELECT DISTINCT c.id, c.title, c."titleEmbedding"
            FROM cvs c
            INNER JOIN recommend_jobs_for_cv r ON c.id = r."cvId"
            WHERE c."titleEmbedding" IS NOT NULL AND c."isMain" = true
        ) c
        ORDER BY RANDOM()
        LIMIT %s
    """

    with db.get_cursor() as cursor:
        cursor.execute(cv_query, (num_cvs,))
        cv_rows = cursor.fetchall()

    results = []

    for cv_row in cv_rows:
        cv_id = cv_row["id"]
        cv_title = cv_row["title"]
        cv_emb = cv_row["titleEmbedding"]
        if isinstance(cv_emb, str):
            cv_emb = json.loads(cv_emb)
        cv_emb = np.array(cv_emb)

        # Get recommended jobs for this CV
        rec_query = """
            SELECT j.title, j."titleEmbedding", r.similarity
            FROM recommend_jobs_for_cv r
            INNER JOIN jobs j ON r."jobId" = j.id
            WHERE r."cvId" = %s AND j."titleEmbedding" IS NOT NULL
            ORDER BY r.similarity DESC
            LIMIT 10
        """
        with db.get_cursor() as cursor:
            cursor.execute(rec_query, (cv_id,))
            rec_jobs = cursor.fetchall()

        # Get random jobs (NOT recommended)
        random_query = """
            SELECT j.title, j."titleEmbedding"
            FROM jobs j
            WHERE j."titleEmbedding" IS NOT NULL
            AND j.id NOT IN (SELECT "jobId" FROM recommend_jobs_for_cv WHERE "cvId" = %s)
            ORDER BY RANDOM()
            LIMIT 10
        """
        with db.get_cursor() as cursor:
            cursor.execute(random_query, (cv_id,))
            random_jobs = cursor.fetchall()

        # Calculate similarities
        rec_sims = []
        for job in rec_jobs:
            job_emb = job["titleEmbedding"]
            if isinstance(job_emb, str):
                job_emb = json.loads(job_emb)
            job_emb = np.array(job_emb)
            sim = np.dot(cv_emb, job_emb) / (np.linalg.norm(cv_emb) * np.linalg.norm(job_emb))
            rec_sims.append(sim)

        random_sims = []
        for job in random_jobs:
            job_emb = job["titleEmbedding"]
            if isinstance(job_emb, str):
                job_emb = json.loads(job_emb)
            job_emb = np.array(job_emb)
            sim = np.dot(cv_emb, job_emb) / (np.linalg.norm(cv_emb) * np.linalg.norm(job_emb))
            random_sims.append(sim)

        if rec_sims and random_sims:
            results.append({
                "cv_title": cv_title,
                "rec_avg": np.mean(rec_sims),
                "rec_max": np.max(rec_sims),
                "random_avg": np.mean(random_sims),
                "random_max": np.max(random_sims),
            })

    if not results:
        logger.warning("No data for recommendation quality visualization")
        return

    # Create comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Average similarity comparison
    cv_names = [truncate_title(r["cv_title"], 20) for r in results]
    rec_avgs = [r["rec_avg"] for r in results]
    random_avgs = [r["random_avg"] for r in results]

    x = np.arange(len(cv_names))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, rec_avgs, width, label='Recommended Jobs', color='#2ecc71', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, random_avgs, width, label='Random Jobs', color='#e74c3c', edgecolor='black')

    axes[0].set_xlabel('CV', fontsize=11)
    axes[0].set_ylabel('Average Cosine Similarity', fontsize=11)
    axes[0].set_title('Recommendation Quality: Recommended vs Random Jobs\n(Higher = Better Match)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cv_names, rotation=45, ha='right', fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, rec_avgs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, random_avgs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Overall comparison
    all_rec_avg = np.mean(rec_avgs)
    all_random_avg = np.mean(random_avgs)
    improvement = ((all_rec_avg - all_random_avg) / all_random_avg) * 100

    bars = axes[1].bar(['Recommended\nJobs', 'Random\nJobs'],
                       [all_rec_avg, all_random_avg],
                       color=['#2ecc71', '#e74c3c'],
                       edgecolor='black', linewidth=2)

    axes[1].set_ylabel('Average Cosine Similarity', fontsize=11)
    axes[1].set_title(f'Overall: Recommendations are {improvement:.1f}% Better\n(Across {len(results)} CVs)',
                     fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, [all_rec_avg, all_random_avg]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")
    logger.info(f"  Recommendation improvement: {improvement:.1f}%")


def visualize_similar_jobs_quality(db: DatabaseConnection, output_path: str, num_seeds: int = 10):
    """Show that similar jobs have HIGHER similarity than random jobs"""

    # Get seed jobs with similar jobs
    seed_query = """
        SELECT j.id, j.title, j."titleEmbedding"
        FROM (
            SELECT DISTINCT j.id, j.title, j."titleEmbedding"
            FROM jobs j
            INNER JOIN similar_jobs s ON j.id = s."jobId"
            WHERE j."titleEmbedding" IS NOT NULL
        ) j
        ORDER BY RANDOM()
        LIMIT %s
    """

    with db.get_cursor() as cursor:
        cursor.execute(seed_query, (num_seeds,))
        seed_rows = cursor.fetchall()

    results = []

    for seed_row in seed_rows:
        seed_id = seed_row["id"]
        seed_title = seed_row["title"]
        seed_emb = seed_row["titleEmbedding"]
        if isinstance(seed_emb, str):
            seed_emb = json.loads(seed_emb)
        seed_emb = np.array(seed_emb)

        # Get similar jobs
        sim_query = """
            SELECT j.title, j."titleEmbedding", s.similarity
            FROM similar_jobs s
            INNER JOIN jobs j ON s."similarJobId" = j.id
            WHERE s."jobId" = %s AND j."titleEmbedding" IS NOT NULL
            ORDER BY s.similarity DESC
            LIMIT 10
        """
        with db.get_cursor() as cursor:
            cursor.execute(sim_query, (seed_id,))
            sim_jobs = cursor.fetchall()

        # Get random jobs (NOT similar)
        random_query = """
            SELECT j.title, j."titleEmbedding"
            FROM jobs j
            WHERE j."titleEmbedding" IS NOT NULL
            AND j.id != %s
            AND j.id NOT IN (SELECT "similarJobId" FROM similar_jobs WHERE "jobId" = %s)
            ORDER BY RANDOM()
            LIMIT 10
        """
        with db.get_cursor() as cursor:
            cursor.execute(random_query, (seed_id, seed_id))
            random_jobs = cursor.fetchall()

        # Calculate similarities
        sim_sims = []
        for job in sim_jobs:
            job_emb = job["titleEmbedding"]
            if isinstance(job_emb, str):
                job_emb = json.loads(job_emb)
            job_emb = np.array(job_emb)
            sim = np.dot(seed_emb, job_emb) / (np.linalg.norm(seed_emb) * np.linalg.norm(job_emb))
            sim_sims.append(sim)

        random_sims = []
        for job in random_jobs:
            job_emb = job["titleEmbedding"]
            if isinstance(job_emb, str):
                job_emb = json.loads(job_emb)
            job_emb = np.array(job_emb)
            sim = np.dot(seed_emb, job_emb) / (np.linalg.norm(seed_emb) * np.linalg.norm(job_emb))
            random_sims.append(sim)

        if sim_sims and random_sims:
            results.append({
                "seed_title": seed_title,
                "similar_avg": np.mean(sim_sims),
                "random_avg": np.mean(random_sims),
            })

    if not results:
        logger.warning("No data for similar jobs quality visualization")
        return

    # Create comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Per-job comparison
    job_names = [truncate_title(r["seed_title"], 18) for r in results]
    sim_avgs = [r["similar_avg"] for r in results]
    random_avgs = [r["random_avg"] for r in results]

    x = np.arange(len(job_names))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, sim_avgs, width, label='Similar Jobs (from system)', color='#3498db', edgecolor='black')
    bars2 = axes[0].bar(x + width/2, random_avgs, width, label='Random Jobs', color='#e74c3c', edgecolor='black')

    axes[0].set_xlabel('Seed Job', fontsize=11)
    axes[0].set_ylabel('Average Cosine Similarity', fontsize=11)
    axes[0].set_title('Similar Jobs Quality: System vs Random\n(Higher = Better Match)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(job_names, rotation=45, ha='right', fontsize=9)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Overall
    all_sim_avg = np.mean(sim_avgs)
    all_random_avg = np.mean(random_avgs)
    improvement = ((all_sim_avg - all_random_avg) / all_random_avg) * 100

    bars = axes[1].bar(['Similar Jobs\n(System)', 'Random\nJobs'],
                       [all_sim_avg, all_random_avg],
                       color=['#3498db', '#e74c3c'],
                       edgecolor='black', linewidth=2)

    axes[1].set_ylabel('Average Cosine Similarity', fontsize=11)
    axes[1].set_title(f'Overall: Similar Jobs are {improvement:.1f}% Better\n(Across {len(results)} seed jobs)',
                     fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, [all_sim_avg, all_random_avg]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")
    logger.info(f"  Similar jobs improvement: {improvement:.1f}%")


def visualize_case_study(db: DatabaseConnection, output_path: str):
    """Show a single CV with its top recommended jobs - easy to understand"""

    # Get one CV with recommendations
    cv_query = """
        SELECT c.id, c.title, c."titleEmbedding"
        FROM cvs c
        INNER JOIN recommend_jobs_for_cv r ON c.id = r."cvId"
        WHERE c."titleEmbedding" IS NOT NULL AND c."isMain" = true
        GROUP BY c.id, c.title, c."titleEmbedding"
        HAVING COUNT(*) >= 5
        ORDER BY RANDOM()
        LIMIT 1
    """

    with db.get_cursor() as cursor:
        cursor.execute(cv_query)
        cv_row = cursor.fetchone()

    if not cv_row:
        logger.warning("No CV found for case study")
        return

    cv_id = cv_row["id"]
    cv_title = cv_row["title"]

    # Get top 5 recommended jobs
    rec_query = """
        SELECT j.title, r.similarity
        FROM recommend_jobs_for_cv r
        INNER JOIN jobs j ON r."jobId" = j.id
        WHERE r."cvId" = %s
        ORDER BY r.similarity DESC
        LIMIT 5
    """

    with db.get_cursor() as cursor:
        cursor.execute(rec_query, (cv_id,))
        rec_jobs = cursor.fetchall()

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    job_titles = [truncate_title(j["title"], 40) for j in rec_jobs]
    similarities = [j["similarity"] for j in rec_jobs]

    colors = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#138d75']
    bars = ax.barh(range(len(job_titles)), similarities, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_yticks(range(len(job_titles)))
    ax.set_yticklabels(job_titles, fontsize=11)
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        ax.text(sim + 0.02, bar.get_y() + bar.get_height()/2,
               f'{sim:.3f}', va='center', fontsize=12, fontweight='bold')

    ax.set_title(f'Case Study: Job Recommendations for CV\n"{truncate_title(cv_title, 60)}"',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add explanation
    ax.text(0.5, -0.12,
           "Higher similarity = Job requirements better match CV skills/experience",
           transform=ax.transAxes, ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {output_path}")


def main():
    output_dir = Path("./visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("VISUALIZATION WITH LABELS")
    logger.info("=" * 60)

    db = DatabaseConnection()

    # 0. NEW: Load jobs by ACTUAL INDUSTRY from database
    logger.info("\n0. Loading jobs by INDUSTRY (from database)...")
    jobs_by_industry = load_jobs_by_industry(db, jobs_per_industry=250, top_industries=12)
    total_industry_jobs = sum(len(jobs) for jobs in jobs_by_industry.values())
    logger.info(f"Total jobs by industry: {total_industry_jobs}")

    if total_industry_jobs >= 5:
        logger.info("Creating job clusters by INDUSTRY visualization...")
        visualize_jobs_by_industry(jobs_by_industry, str(output_dir / "jobs_by_industry.png"))

        # NEW: Industry similarity heatmap
        logger.info("\nCreating industry similarity heatmap...")
        visualize_industry_similarity_heatmap(jobs_by_industry, str(output_dir / "industry_heatmap.png"))

    # NEW: Recommendation quality comparison (Recommended vs Random)
    logger.info("\n=== RECOMMENDATION QUALITY VISUALIZATION ===")
    logger.info("Comparing recommended jobs vs random jobs for CVs...")
    visualize_recommendation_quality(db, str(output_dir / "recommendation_quality.png"), num_cvs=10)

    # NEW: Similar jobs quality comparison
    logger.info("\nComparing similar jobs vs random jobs...")
    visualize_similar_jobs_quality(db, str(output_dir / "similar_jobs_quality.png"), num_seeds=10)

    # NEW: Case study - single CV with recommendations
    logger.info("\nCreating case study visualization...")
    visualize_case_study(db, str(output_dir / "case_study_cv.png"))

    # 1. Load sample jobs by category (keyword-based)
    logger.info("\nLoading sample jobs by category (keyword-based)...")
    jobs_by_cat = load_sample_jobs_by_category(db)

    total_jobs = sum(len(jobs) for jobs in jobs_by_cat.values())
    logger.info(f"Total jobs loaded: {total_jobs}")

    if total_jobs < 5:
        logger.error("Not enough jobs to visualize!")
        return

    # Flatten for other uses
    all_jobs = []
    for jobs in jobs_by_cat.values():
        all_jobs.extend(jobs)
    job_ids = [j["id"] for j in all_jobs]

    # 2. Create job clusters visualization with labels
    logger.info("\n1. Creating job clusters with labels...")
    visualize_jobs_with_labels(jobs_by_cat, str(output_dir / "jobs_with_labels.png"))

    # 3. NEW: Load seed jobs and their similar jobs
    logger.info("\n2. Loading seed jobs and their similar jobs...")
    seed_jobs, similar_jobs, similarities = load_seed_jobs_and_their_similars(db, num_seeds=10, similars_per_seed=10)
    logger.info(f"Found {len(seed_jobs)} seed jobs with {len(similar_jobs)} similar jobs, {len(similarities)} connections")

    if seed_jobs and similar_jobs:
        logger.info("Creating seed-similar jobs visualization...")
        visualize_seed_and_similar_jobs(seed_jobs, similar_jobs, similarities,
                                        str(output_dir / "similar_jobs_with_labels.png"))

    # 4. Load CVs and their recommended jobs (CLEAN version)
    logger.info("\n3. Loading CVs with their recommended jobs...")
    cvs, rec_jobs, recommendations = load_cvs_with_their_recommended_jobs(db, num_cvs=8, jobs_per_cv=5)
    logger.info(f"Found {len(cvs)} CVs with {len(rec_jobs)} recommended jobs, {len(recommendations)} connections")

    if cvs and rec_jobs:
        logger.info("Creating CV-Job recommendations visualization (clean)...")
        visualize_cv_job_recommendations_clean(cvs, rec_jobs, recommendations,
                                               str(output_dir / "cv_job_with_labels.png"))

    logger.info("\n" + "=" * 60)
    logger.info("DONE! Check the visualizations folder:")
    logger.info(f"  {output_dir.absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
