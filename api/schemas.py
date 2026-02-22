from pydantic import BaseModel
from typing import Optional


class TextRequest(BaseModel):
    text: str


class FusionRequest(BaseModel):
    text: Optional[str] = None
