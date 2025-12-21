"""
Visualize recommendation results using t-SNE/UMAP for embedding clustering.
Shows CVs and Jobs in 2D space with recommendation connections.

Usage:
    python scripts/visualize_recommendations.py --jobs 500 --cvs 100
    python scripts/visualize_recommendations.py --jobs 1000 --cvs 200 --method umap
"""
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

from recommend_service.database.connection import DatabaseConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_jobs_with_embeddings(db: DatabaseConnection, limit: int = 500):
    """Load jobs with their embeddings (mix of recommended and all jobs)"""
    # First get jobs that have recommendations, then fill with other jobs
    query = """
        (SELECT DISTINCT j.id, j.title, j."titleEmbedding", 1 as priority
        FROM jobs j
        INNER JOIN recommend_jobs_for_cv r ON j.id = r."jobId"
        WHERE j."titleEmbedding" IS NOT NULL)
        UNION ALL
        (SELECT j.id, j.title, j."titleEmbedding", 2 as priority
        FROM jobs j
        WHERE j."titleEmbedding" IS NOT NULL
        AND j.id NOT IN (SELECT DISTINCT "jobId" FROM recommend_jobs_for_cv))
        ORDER BY priority
        LIMIT %s
    """
    with db.get_cursor() as cursor:
        cursor.execute(query, (limit,))
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
                "embedding": np.array(emb)
            })
    return jobs


def load_cvs_with_embeddings(db: DatabaseConnection, limit: int = 100):
    """Load CVs with their embeddings (prioritize CVs that have recommendations)"""
    query = """
        SELECT DISTINCT c.id, c.title, c."titleEmbedding"
        FROM cvs c
        INNER JOIN recommend_jobs_for_cv r ON c.id = r."cvId"
        WHERE c."titleEmbedding" IS NOT NULL AND c."isMain" = true
        LIMIT %s
    """
    with db.get_cursor() as cursor:
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()

    cvs = []
    for row in rows:
        emb = row["titleEmbedding"]
        if isinstance(emb, str):
            emb = json.loads(emb)
        if emb:
            cvs.append({
                "id": row["id"],
                "title": row["title"],
                "embedding": np.array(emb)
            })
    return cvs


def load_recommendations(db: DatabaseConnection, cv_ids: list, job_ids: list):
    """Load recommendations between CVs and Jobs"""
    if not cv_ids or not job_ids:
        return []

    query = """
        SELECT "cvId", "jobId", similarity
        FROM recommend_jobs_for_cv
        WHERE "cvId" = ANY(%s) AND "jobId" = ANY(%s)
        ORDER BY similarity DESC
    """
    with db.get_cursor() as cursor:
        cursor.execute(query, (cv_ids, job_ids))
        return cursor.fetchall()


def load_similar_jobs(db: DatabaseConnection, job_ids: list):
    """Load similar job pairs"""
    if not job_ids:
        return []

    query = """
        SELECT "jobId", "similarJobId", similarity
        FROM similar_jobs
        WHERE "jobId" = ANY(%s) AND "similarJobId" = ANY(%s)
    """
    with db.get_cursor() as cursor:
        cursor.execute(query, (job_ids, job_ids))
        return cursor.fetchall()


def categorize_jobs(jobs: list) -> dict:
    """Simple categorization based on job title keywords"""
    categories = {
        "IT/Software": ["developer", "engineer", "software", "backend", "frontend", "fullstack", "devops", "data", "ai", "ml", "python", "java", "react", "nodejs"],
        "Marketing": ["marketing", "seo", "content", "social media", "digital", "brand"],
        "Sales": ["sales", "business", "account", "customer"],
        "HR": ["hr", "human resource", "recruitment", "talent"],
        "Finance": ["finance", "accountant", "accounting", "audit"],
        "Design": ["design", "ui", "ux", "graphic", "creative"],
        "Management": ["manager", "director", "lead", "head", "supervisor"],
        "Other": []
    }

    job_categories = {}
    for job in jobs:
        title_lower = job["title"].lower() if job["title"] else ""
        assigned = False
        for cat, keywords in categories.items():
            if cat == "Other":
                continue
            for kw in keywords:
                if kw in title_lower:
                    job_categories[job["id"]] = cat
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            job_categories[job["id"]] = "Other"

    return job_categories


def reduce_dimensions(embeddings: np.ndarray, method: str = "tsne", perplexity: int = 30):
    """Reduce embedding dimensions to 2D"""
    if method == "tsne":
        # Adjust perplexity based on sample size
        n_samples = embeddings.shape[0]
        adjusted_perplexity = min(perplexity, max(5, n_samples // 3))

        logger.info(f"Running t-SNE with perplexity={adjusted_perplexity} on {n_samples} samples...")
        reducer = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42, max_iter=1000)
        return reducer.fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap
            logger.info(f"Running UMAP on {embeddings.shape[0]} samples...")
            reducer = umap.UMAP(n_components=2, random_state=42)
            return reducer.fit_transform(embeddings)
        except ImportError:
            logger.warning("UMAP not installed. Falling back to t-SNE. Install with: pip install umap-learn")
            return reduce_dimensions(embeddings, method="tsne", perplexity=perplexity)
    else:
        raise ValueError(f"Unknown method: {method}")


def plot_job_clusters(jobs: list, coords_2d: np.ndarray, job_categories: dict, output_path: str):
    """Plot job clusters by category"""
    plt.figure(figsize=(14, 10))

    # Color map for categories
    category_colors = {
        "IT/Software": "#2ecc71",
        "Marketing": "#e74c3c",
        "Sales": "#3498db",
        "HR": "#9b59b6",
        "Finance": "#f39c12",
        "Design": "#e91e63",
        "Management": "#00bcd4",
        "Other": "#95a5a6"
    }

    # Group jobs by category
    category_points = defaultdict(list)
    category_indices = defaultdict(list)

    for i, job in enumerate(jobs):
        cat = job_categories.get(job["id"], "Other")
        category_points[cat].append(coords_2d[i])
        category_indices[cat].append(i)

    # Plot each category
    for cat, points in category_points.items():
        if points:
            points = np.array(points)
            color = category_colors.get(cat, "#95a5a6")
            plt.scatter(points[:, 0], points[:, 1],
                       c=color, label=f"{cat} ({len(points)})",
                       alpha=0.6, s=50, edgecolors='white', linewidth=0.5)

    plt.title("Job Clusters by Category (t-SNE)", fontsize=14, fontweight='bold')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved job clusters plot to: {output_path}")


def plot_cv_job_recommendations(cvs: list, jobs: list, cv_coords: np.ndarray, job_coords: np.ndarray,
                                 recommendations: list, output_path: str, top_k: int = 3):
    """Plot CVs and Jobs with recommendation connections"""
    plt.figure(figsize=(16, 12))

    # Create ID to index mapping
    cv_id_to_idx = {cv["id"]: i for i, cv in enumerate(cvs)}
    job_id_to_idx = {job["id"]: i for i, job in enumerate(jobs)}

    # Plot jobs (blue)
    plt.scatter(job_coords[:, 0], job_coords[:, 1],
               c='#3498db', label=f'Jobs ({len(jobs)})',
               alpha=0.4, s=30, marker='o')

    # Plot CVs (red, larger)
    plt.scatter(cv_coords[:, 0], cv_coords[:, 1],
               c='#e74c3c', label=f'CVs ({len(cvs)})',
               alpha=0.8, s=100, marker='s', edgecolors='white', linewidth=1)

    # Draw recommendation lines (top-k per CV)
    cv_recs = defaultdict(list)
    for rec in recommendations:
        cv_id = rec["cvId"]
        job_id = rec["jobId"]
        if cv_id in cv_id_to_idx and job_id in job_id_to_idx:
            cv_recs[cv_id].append((job_id, rec["similarity"]))

    lines_drawn = 0
    for cv_id, recs in cv_recs.items():
        # Sort by similarity and take top-k
        recs.sort(key=lambda x: x[1], reverse=True)
        for job_id, sim in recs[:top_k]:
            cv_idx = cv_id_to_idx[cv_id]
            job_idx = job_id_to_idx[job_id]

            # Draw line with alpha based on similarity
            alpha = min(0.8, sim * 0.8)
            plt.plot([cv_coords[cv_idx, 0], job_coords[job_idx, 0]],
                    [cv_coords[cv_idx, 1], job_coords[job_idx, 1]],
                    'g-', alpha=alpha, linewidth=0.5)
            lines_drawn += 1

    plt.title(f"CV-Job Recommendations (top-{top_k} per CV, {lines_drawn} connections)",
             fontsize=14, fontweight='bold')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CV-Job recommendations plot to: {output_path}")


def plot_similar_jobs_network(jobs: list, job_coords: np.ndarray, similar_jobs: list,
                               job_categories: dict, output_path: str, top_k: int = 3):
    """Plot similar jobs network"""
    plt.figure(figsize=(14, 10))

    # Color map for categories
    category_colors = {
        "IT/Software": "#2ecc71",
        "Marketing": "#e74c3c",
        "Sales": "#3498db",
        "HR": "#9b59b6",
        "Finance": "#f39c12",
        "Design": "#e91e63",
        "Management": "#00bcd4",
        "Other": "#95a5a6"
    }

    job_id_to_idx = {job["id"]: i for i, job in enumerate(jobs)}

    # Group and sort similar jobs
    job_similar = defaultdict(list)
    for sj in similar_jobs:
        job_id = sj["jobId"]
        similar_id = sj["similarJobId"]
        if job_id in job_id_to_idx and similar_id in job_id_to_idx:
            job_similar[job_id].append((similar_id, sj["similarity"]))

    # Draw similarity lines first (so they're behind points)
    lines_drawn = 0
    for job_id, similars in job_similar.items():
        similars.sort(key=lambda x: x[1], reverse=True)
        for similar_id, sim in similars[:top_k]:
            idx1 = job_id_to_idx[job_id]
            idx2 = job_id_to_idx[similar_id]

            alpha = min(0.6, sim * 0.6)
            plt.plot([job_coords[idx1, 0], job_coords[idx2, 0]],
                    [job_coords[idx1, 1], job_coords[idx2, 1]],
                    '-', color='#bdc3c7', alpha=alpha, linewidth=0.3)
            lines_drawn += 1

    # Plot jobs by category
    category_points = defaultdict(list)
    for i, job in enumerate(jobs):
        cat = job_categories.get(job["id"], "Other")
        category_points[cat].append(job_coords[i])

    for cat, points in category_points.items():
        if points:
            points = np.array(points)
            color = category_colors.get(cat, "#95a5a6")
            plt.scatter(points[:, 0], points[:, 1],
                       c=color, label=f"{cat} ({len(points)})",
                       alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

    plt.title(f"Similar Jobs Network (top-{top_k} per job, {lines_drawn} connections)",
             fontsize=14, fontweight='bold')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved similar jobs network plot to: {output_path}")


def plot_combined_visualization(cvs: list, jobs: list, all_coords: np.ndarray,
                                 recommendations: list, job_categories: dict, output_path: str):
    """Combined visualization with CVs, Jobs, and recommendations"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    n_cvs = len(cvs)
    cv_coords = all_coords[:n_cvs]
    job_coords = all_coords[n_cvs:]

    # Color map for categories
    category_colors = {
        "IT/Software": "#2ecc71",
        "Marketing": "#e74c3c",
        "Sales": "#3498db",
        "HR": "#9b59b6",
        "Finance": "#f39c12",
        "Design": "#e91e63",
        "Management": "#00bcd4",
        "Other": "#95a5a6"
    }

    cv_id_to_idx = {cv["id"]: i for i, cv in enumerate(cvs)}
    job_id_to_idx = {job["id"]: i for i, job in enumerate(jobs)}

    # === Left plot: All points with categories ===
    ax1 = axes[0]

    # Plot jobs by category
    category_points = defaultdict(list)
    for i, job in enumerate(jobs):
        cat = job_categories.get(job["id"], "Other")
        category_points[cat].append(job_coords[i])

    for cat, points in category_points.items():
        if points:
            points = np.array(points)
            color = category_colors.get(cat, "#95a5a6")
            ax1.scatter(points[:, 0], points[:, 1],
                       c=color, label=f"{cat}",
                       alpha=0.5, s=30, marker='o')

    # Plot CVs
    ax1.scatter(cv_coords[:, 0], cv_coords[:, 1],
               c='black', label=f'CVs ({n_cvs})',
               alpha=0.9, s=120, marker='*', edgecolors='white', linewidth=1)

    ax1.set_title("CVs and Jobs in Embedding Space", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # === Right plot: Recommendations ===
    ax2 = axes[1]

    # Plot jobs (light blue)
    ax2.scatter(job_coords[:, 0], job_coords[:, 1],
               c='#3498db', alpha=0.3, s=25, marker='o', label=f'Jobs ({len(jobs)})')

    # Draw recommendation lines
    cv_recs = defaultdict(list)
    for rec in recommendations:
        cv_id = rec["cvId"]
        job_id = rec["jobId"]
        if cv_id in cv_id_to_idx and job_id in job_id_to_idx:
            cv_recs[cv_id].append((job_id, rec["similarity"]))

    # Draw lines for top-5 recommendations per CV
    for cv_id, recs in cv_recs.items():
        recs.sort(key=lambda x: x[1], reverse=True)
        for job_id, sim in recs[:5]:
            cv_idx = cv_id_to_idx[cv_id]
            job_idx = job_id_to_idx[job_id]

            alpha = min(0.7, sim * 0.7)
            ax2.plot([cv_coords[cv_idx, 0], job_coords[job_idx, 0]],
                    [cv_coords[cv_idx, 1], job_coords[job_idx, 1]],
                    '-', color='#27ae60', alpha=alpha, linewidth=0.8)

    # Plot CVs (on top)
    ax2.scatter(cv_coords[:, 0], cv_coords[:, 1],
               c='#e74c3c', label=f'CVs ({n_cvs})',
               alpha=0.9, s=120, marker='s', edgecolors='white', linewidth=1.5)

    ax2.set_title("CV-Job Recommendations (top-5 per CV)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Job Recommendation System Visualization", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved combined visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize recommendation results")
    parser.add_argument("--jobs", type=int, default=500, help="Number of jobs to visualize")
    parser.add_argument("--cvs", type=int, default=100, help="Number of CVs to visualize")
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne", help="Dimension reduction method")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Output directory for plots")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("RECOMMENDATION VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"Jobs: {args.jobs}, CVs: {args.cvs}, Method: {args.method}")

    # Load data
    db = DatabaseConnection()

    logger.info("Loading jobs...")
    jobs = load_jobs_with_embeddings(db, limit=args.jobs)
    logger.info(f"Loaded {len(jobs)} jobs with embeddings")

    logger.info("Loading CVs...")
    cvs = load_cvs_with_embeddings(db, limit=args.cvs)
    logger.info(f"Loaded {len(cvs)} CVs with embeddings")

    if not jobs:
        logger.error("No jobs with embeddings found!")
        return

    # Categorize jobs
    job_categories = categorize_jobs(jobs)
    category_counts = defaultdict(int)
    for cat in job_categories.values():
        category_counts[cat] += 1
    logger.info(f"Job categories: {dict(category_counts)}")

    # Prepare embeddings
    job_embeddings = np.array([j["embedding"] for j in jobs])

    # 1. Plot job clusters only
    logger.info("Creating job clusters visualization...")
    job_coords = reduce_dimensions(job_embeddings, method=args.method)
    plot_job_clusters(jobs, job_coords, job_categories,
                      str(output_dir / "job_clusters.png"))

    # 2. Load similar jobs and plot network
    job_ids = [j["id"] for j in jobs]
    similar_jobs = load_similar_jobs(db, job_ids)
    logger.info(f"Loaded {len(similar_jobs)} similar job pairs")

    if similar_jobs:
        logger.info("Creating similar jobs network visualization...")
        plot_similar_jobs_network(jobs, job_coords, similar_jobs, job_categories,
                                  str(output_dir / "similar_jobs_network.png"))

    # 3. If we have CVs, create CV-Job visualizations
    if cvs:
        cv_embeddings = np.array([c["embedding"] for c in cvs])

        # Combine embeddings for joint visualization
        all_embeddings = np.vstack([cv_embeddings, job_embeddings])
        logger.info(f"Reducing dimensions for {len(all_embeddings)} combined embeddings...")
        all_coords = reduce_dimensions(all_embeddings, method=args.method)

        cv_coords = all_coords[:len(cvs)]
        job_coords_combined = all_coords[len(cvs):]

        # Load recommendations
        cv_ids = [c["id"] for c in cvs]
        recommendations = load_recommendations(db, cv_ids, job_ids)
        logger.info(f"Loaded {len(recommendations)} recommendations")

        # Plot CV-Job recommendations
        if recommendations:
            logger.info("Creating CV-Job recommendations visualization...")
            plot_cv_job_recommendations(cvs, jobs, cv_coords, job_coords_combined,
                                        recommendations, str(output_dir / "cv_job_recommendations.png"))

        # Combined visualization
        logger.info("Creating combined visualization...")
        plot_combined_visualization(cvs, jobs, all_coords, recommendations, job_categories,
                                    str(output_dir / "combined_visualization.png"))

    logger.info("=" * 60)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
