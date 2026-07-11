import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoscaleConfig:
    min_workers: int
    max_workers: int
    keep_free_gb: float
    per_job_gb: float
    job_stagger_seconds: float = 0.0  # delay between *new* workers


class LiveAutoscaler:
    def __init__(self, cfg: AutoscaleConfig) -> None:
        self.cfg = cfg
        # Time of last *increase* in peak concurrency
        self.last_new_worker_launch_ts: float = 0.0
        # Highest concurrent worker count we have seen so far
        self.peak_workers_seen: int = 0

    def should_launch(
        self,
        running_workers: int,
        target_workers: int,
        now: Optional[float] = None,
    ) -> bool:
        """
        Decide whether to launch *one* more worker.

        - Replacement workers (keeping concurrency constant) are launched
          immediately.
        - Only when increasing peak concurrency do we respect job_stagger_seconds.
        """
        if now is None:
            now = time.monotonic()

        # No reason to launch if we're already at/above the desired level.
        if running_workers >= target_workers:
            return False

        # Replacement: some worker died or finished, but we're not increasing
        # peak concurrency. Start the replacement immediately.
        if running_workers < self.peak_workers_seen:
            return True

        # New parallel worker: we are about to increase peak concurrency.
        stagger = self.cfg.job_stagger_seconds
        if stagger > 0.0 and (now - self.last_new_worker_launch_ts) < stagger:
            return False

        # Allow this increase, and record the new peak.
        self.last_new_worker_launch_ts = now
        self.peak_workers_seen = running_workers + 1
        return True

