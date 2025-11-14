"""
File-based Task Storage for Process Pool Workers
Uses JSON files to share task status between API server and worker processes
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from loguru import logger


class TaskStorage:
    """File-based task storage for multi-process communication"""

    def __init__(self, storage_dir: Path = None):
        """
        Initialize task storage

        Args:
            storage_dir: Directory to store task status files
        """
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent / "workspace" / "tasks"

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📂 Task storage initialized: {self.storage_dir}")

    def _get_task_file(self, task_id: str) -> Path:
        """Get path to task status file"""
        return self.storage_dir / f"{task_id}.json"

    def create(self, task_id: str, initial_data: Optional[Dict] = None) -> None:
        """
        Create a new task

        Args:
            task_id: Unique task ID
            initial_data: Initial task data
        """
        data = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "Task queued",
            "created_at": datetime.now().isoformat(),
            **(initial_data or {})
        }

        task_file = self._get_task_file(task_id)
        with open(task_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.debug(f"📝 Task created: {task_id}")

    def get(self, task_id: str) -> Optional[Dict]:
        """
        Get task status

        Args:
            task_id: Task ID

        Returns:
            Task data or None if not found
        """
        task_file = self._get_task_file(task_id)

        if not task_file.exists():
            return None

        try:
            with open(task_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading task {task_id}: {e}")
            return None

    def update(self, task_id: str, data: Dict) -> None:
        """
        Update task status

        Args:
            task_id: Task ID
            data: Data to update (will be merged with existing)
        """
        task_file = self._get_task_file(task_id)

        # Read existing data
        existing = self.get(task_id) or {}

        # Merge with new data
        existing.update(data)
        existing["updated_at"] = datetime.now().isoformat()

        # Write back
        with open(task_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.debug(f"📝 Task updated: {task_id} - {data.get('status', 'unknown')}")

    def complete(self, task_id: str, result: Dict) -> None:
        """
        Mark task as completed

        Args:
            task_id: Task ID
            result: Result data
        """
        self.update(task_id, {
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"✅ Task completed: {task_id}")

    def fail(self, task_id: str, error: str) -> None:
        """
        Mark task as failed

        Args:
            task_id: Task ID
            error: Error message
        """
        self.update(task_id, {
            "status": "failed",
            "error": error,
            "completed_at": datetime.now().isoformat()
        })
        logger.error(f"❌ Task failed: {task_id} - {error}")

    def delete(self, task_id: str) -> None:
        """
        Delete task file

        Args:
            task_id: Task ID
        """
        task_file = self._get_task_file(task_id)
        if task_file.exists():
            task_file.unlink()
            logger.debug(f"🗑️ Task deleted: {task_id}")

    def list_all(self) -> list[Dict]:
        """
        List all tasks

        Returns:
            List of task data
        """
        tasks = []
        for task_file in self.storage_dir.glob("*.json"):
            try:
                with open(task_file, 'r') as f:
                    tasks.append(json.load(f))
            except Exception as e:
                logger.error(f"Error reading {task_file}: {e}")

        return sorted(tasks, key=lambda x: x.get('created_at', ''), reverse=True)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """
        Clean up old task files

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of files deleted
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted = 0

        for task_file in self.storage_dir.glob("*.json"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(task_file.stat().st_mtime)
                if mtime < cutoff:
                    task_file.unlink()
                    deleted += 1
            except Exception as e:
                logger.error(f"Error cleaning up {task_file}: {e}")

        if deleted > 0:
            logger.info(f"🗑️ Cleaned up {deleted} old task file(s)")

        return deleted


# Global instance
_storage = None

def get_storage() -> TaskStorage:
    """Get global task storage instance"""
    global _storage
    if _storage is None:
        _storage = TaskStorage()
    return _storage
