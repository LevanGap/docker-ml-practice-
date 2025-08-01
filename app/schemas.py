from typing import List
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)
