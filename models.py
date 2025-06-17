from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocConfig(BaseModel):
    delimiter: str
    stdNm: str
    position: int
    token: Optional[str] = None

class TemplateConfig(BaseModel):
    stopWords: str
    columnLength: int
    delimiter: str
    invoiceLine: int

class RemittanceConfig(BaseModel):
    delimiter: str
    format: Optional[str] = None
    stdNm: str
    position: int
    token: str

class ExtractionConfig(BaseModel):
    templateNm: str
    docConfig: List[DocConfig]
    templateConfig: TemplateConfig
    footer: List[str]
    header: List[str]
    remittanceConfig: List[RemittanceConfig]

    