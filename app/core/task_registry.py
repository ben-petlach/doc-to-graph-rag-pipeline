from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional


@dataclass(slots=True)
class FileRecord:
    name: str
    stage: str
    error: Optional[str] = None


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    status: str  # queued|processing|completed|failed
    processed_files: int = 0
    total_files: int = 0
    error: Optional[str] = None

    @property
    def progress(self) -> str:
        if self.total_files <= 0:
            return "0/0 files"
        return f"{self.processed_files}/{self.total_files} files"


class InMemoryRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._files: dict[str, FileRecord] = {}
        self._tasks: dict[str, TaskRecord] = {}

    def ensure_file_from_disk(self, *, filename: str) -> None:
        with self._lock:
            if filename not in self._files:
                self._files[filename] = FileRecord(name=filename, stage="raw")

    def sync_files_from_disk(self, *, data_dir: Path) -> None:
        if not data_dir.exists():
            return

        disk_files = [p.name for p in data_dir.iterdir() if p.is_file()]
        with self._lock:
            for name in disk_files:
                if name not in self._files:
                    self._files[name] = FileRecord(name=name, stage="raw")

            for name in list(self._files.keys()):
                if name not in disk_files:
                    # keep record if task/job wants it, but mark as missing
                    if self._files[name].stage != "deleted":
                        self._files[name].stage = "missing"

    def list_files(self) -> list[FileRecord]:
        with self._lock:
            return [FileRecord(name=r.name, stage=r.stage, error=r.error) for r in self._files.values()]

    def set_file_stage(self, *, filename: str, stage: str, error: Optional[str] = None) -> None:
        with self._lock:
            rec = self._files.get(filename) or FileRecord(name=filename, stage=stage)
            rec.stage = stage
            rec.error = error
            self._files[filename] = rec

    def delete_file_record(self, *, filename: str) -> None:
        with self._lock:
            if filename in self._files:
                self._files[filename].stage = "deleted"

    def create_task(self, *, task_id: str, total_files: int) -> TaskRecord:
        rec = TaskRecord(task_id=task_id, status="queued", processed_files=0, total_files=total_files)
        with self._lock:
            self._tasks[task_id] = rec
        return rec

    def get_task(self, *, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                return None
            return TaskRecord(
                task_id=rec.task_id,
                status=rec.status,
                processed_files=rec.processed_files,
                total_files=rec.total_files,
                error=rec.error,
            )

    def update_task(
        self,
        *,
        task_id: str,
        status: Optional[str] = None,
        processed_files: Optional[int] = None,
        total_files: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None:
                rec = TaskRecord(task_id=task_id, status="queued")
                self._tasks[task_id] = rec

            if status is not None:
                rec.status = status
            if processed_files is not None:
                rec.processed_files = processed_files
            if total_files is not None:
                rec.total_files = total_files
            if error is not None:
                rec.error = error

