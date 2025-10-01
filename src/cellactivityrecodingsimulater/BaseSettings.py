from pathlib import Path
from typing import Optional
import json

class BaseSettings():
    
    def __init__(self, data: dict):
        self.data = data

    def to_dict(self):
        return self.data
