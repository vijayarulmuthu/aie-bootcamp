from enum import Enum

class TaskStatus(str, Enum):
    """Enum for task status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed" 