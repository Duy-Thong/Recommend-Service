import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from recommend_service.config import settings
from recommend_service.services.recommendation import RecommendationService

logger = logging.getLogger(__name__)


class RecommendationScheduler:
    def __init__(self):
        self.scheduler = BlockingScheduler()
        self.recommendation_service = RecommendationService()
        self.interval_hours = settings.schedule_interval_hours

    def _run_recommendation_job(self) -> None:
        """Execute the recommendation pipeline"""
        logger.info(f"[{datetime.now()}] Starting scheduled recommendation job")
        try:
            stats = self.recommendation_service.run()
            logger.info(f"[{datetime.now()}] Recommendation job completed: {stats}")
        except Exception as e:
            logger.error(f"[{datetime.now()}] Recommendation job failed: {e}", exc_info=True)

    def start(self, run_immediately: bool = True) -> None:
        """
        Start the scheduler.

        Args:
            run_immediately: If True, run the job immediately before starting the schedule
        """
        logger.info(f"Starting scheduler with interval: {self.interval_hours} hours")

        # Run immediately if requested
        if run_immediately:
            logger.info("Running initial recommendation job...")
            self._run_recommendation_job()

        # Schedule the job
        self.scheduler.add_job(
            self._run_recommendation_job,
            trigger=IntervalTrigger(hours=self.interval_hours),
            id="recommendation_job",
            name="Generate job recommendations for CVs",
            replace_existing=True
        )

        logger.info("Scheduler started. Press Ctrl+C to exit.")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")
            self.scheduler.shutdown()

    def run_once(self) -> dict:
        """Run the recommendation job once without scheduling"""
        logger.info("Running recommendation job once...")
        return self.recommendation_service.run()
