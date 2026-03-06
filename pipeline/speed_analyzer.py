import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class TimingRecord:
    file: str
    page: int
    stage: str
    duration_seconds: float


class SpeedTracker:
    """Tracks per-page, per-stage timing data across the pipeline."""

    def __init__(self):
        self._records: list[TimingRecord] = []
        self._active_start: Optional[float] = None
        self._active_file: Optional[str] = None
        self._active_page: Optional[int] = None
        self._active_stage: Optional[str] = None

    def start(self, file: str, page: int, stage: str) -> None:
        """Begin timing a stage for a specific page."""
        if self._active_start is not None:
            logging.warning(
                f"SpeedTracker: overwriting active timer "
                f"({self._active_file} p{self._active_page} {self._active_stage})"
            )
        self._active_file = file
        self._active_page = page
        self._active_stage = stage
        self._active_start = time.perf_counter()

    def stop(self) -> float:
        """Stop the active timer and record it. Returns elapsed seconds."""
        if self._active_start is None:
            logging.warning("SpeedTracker: stop() called with no active timer")
            return 0.0

        elapsed = time.perf_counter() - self._active_start
        self._records.append(TimingRecord(
            file=self._active_file,
            page=self._active_page,
            stage=self._active_stage,
            duration_seconds=round(elapsed, 4),
        ))
        self._active_start = None
        self._active_file = None
        self._active_page = None
        self._active_stage = None
        return elapsed

    def record(self, file: str, page: int, stage: str, duration: float) -> None:
        """Directly add a timing record (useful when timing is managed externally)."""
        self._records.append(TimingRecord(
            file=file, page=page, stage=stage, duration_seconds=round(duration, 4)
        ))

    @property
    def records(self) -> list[TimingRecord]:
        return list(self._records)

    def summary(self) -> str:
        """Return a formatted summary of timing data."""
        if not self._records:
            return "No timing data recorded."

        lines = []
        lines.append(f"{'File':<40} {'Page':>5} {'Stage':<20} {'Duration (s)':>12}")
        lines.append("-" * 80)

        total = 0.0
        stage_totals: dict[str, float] = {}
        stage_counts: dict[str, int] = {}

        for r in self._records:
            lines.append(f"{r.file:<40} {r.page:>5} {r.stage:<20} {r.duration_seconds:>12.4f}")
            total += r.duration_seconds
            stage_totals[r.stage] = stage_totals.get(r.stage, 0.0) + r.duration_seconds
            stage_counts[r.stage] = stage_counts.get(r.stage, 0) + 1

        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<40} {'':>5} {'':20} {total:>12.4f}")
        lines.append("")
        lines.append("Averages by stage:")
        for stage in sorted(stage_totals.keys()):
            avg = stage_totals[stage] / stage_counts[stage]
            lines.append(f"  {stage:<20} avg={avg:.4f}s  total={stage_totals[stage]:.4f}s  count={stage_counts[stage]}")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save timing data as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "page", "stage", "duration_seconds"])
            for r in self._records:
                writer.writerow([r.file, r.page, r.stage, r.duration_seconds])

        logging.info(f"Speed report saved to {path}")


# Global tracker instance for use across pipeline modules
tracker = SpeedTracker()
