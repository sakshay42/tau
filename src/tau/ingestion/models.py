from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str

    title: Optional[str] = None
    text: str

    source: str

    published_at: Optional[datetime] = None
    ingested_at: datetime

    url: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)