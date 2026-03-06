from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional
from uuid import uuid4


@dataclass(slots=True)
class FileRecord:
    id: str
    name: str
    stage: str
    date_uploaded: str
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
        self._files: dict[str, FileRecord] = {}  # key: file_id
        self._files_by_name: dict[str, str] = {}  # key: filename, value: file_id
        self._tasks: dict[str, TaskRecord] = {}

    def ensure_file_from_disk(self, *, filename: str) -> str:
        """Ensure file exists in registry, return its ID."""
        with self._lock:
            if filename in self._files_by_name:
                return self._files_by_name[filename]
            
            file_id = str(uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            self._files[file_id] = FileRecord(
                id=file_id,
                name=filename,
                stage="raw",
                date_uploaded=timestamp
            )
            self._files_by_name[filename] = file_id
            return file_id

    def sync_files_from_disk(self, *, data_dir: Path) -> None:
        if not data_dir.exists():
            return

        disk_files = [p.name for p in data_dir.iterdir() if p.is_file()]
        with self._lock:
            for name in disk_files:
                if name not in self._files_by_name:
                    file_id = str(uuid4())
                    timestamp = datetime.now(timezone.utc).isoformat()
                    self._files[file_id] = FileRecord(
                        id=file_id,
                        name=name,
                        stage="raw",
                        date_uploaded=timestamp
                    )
                    self._files_by_name[name] = file_id

            for name in list(self._files_by_name.keys()):
                if name not in disk_files:
                    file_id = self._files_by_name[name]
                    # keep record if task/job wants it, but mark as missing
                    if self._files[file_id].stage != "deleted":
                        self._files[file_id].stage = "missing"

    def list_files(self) -> list[FileRecord]:
        with self._lock:
            return [
                FileRecord(
                    id=r.id,
                    name=r.name,
                    stage=r.stage,
                    date_uploaded=r.date_uploaded,
                    error=r.error
                )
                for r in self._files.values()
            ]

    def set_file_stage(self, *, filename: str, stage: str, error: Optional[str] = None) -> str:
        """Set file stage, return file ID."""
        with self._lock:
            if filename in self._files_by_name:
                file_id = self._files_by_name[filename]
                rec = self._files[file_id]
            else:
                file_id = str(uuid4())
                timestamp = datetime.now(timezone.utc).isoformat()
                rec = FileRecord(
                    id=file_id,
                    name=filename,
                    stage=stage,
                    date_uploaded=timestamp
                )
                self._files[file_id] = rec
                self._files_by_name[filename] = file_id
            
            rec.stage = stage
            rec.error = error
            return file_id

    def delete_file_record(self, *, filename: str) -> None:
        """Delete file record by filename (deprecated - use delete_file_by_id)."""
        with self._lock:
            if filename in self._files_by_name:
                file_id = self._files_by_name[filename]
                self._files[file_id].stage = "deleted"

    def get_file_by_id(self, *, file_id: str) -> Optional[FileRecord]:
        """Get file record by ID."""
        with self._lock:
            rec = self._files.get(file_id)
            if rec is None:
                return None
            return FileRecord(
                id=rec.id,
                name=rec.name,
                stage=rec.stage,
                date_uploaded=rec.date_uploaded,
                error=rec.error
            )

    def delete_file_by_id(self, *, file_id: str) -> bool:
        """Delete file record by ID. Returns True if found, False otherwise."""
        with self._lock:
            if file_id in self._files:
                self._files[file_id].stage = "deleted"
                return True
            return False

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

