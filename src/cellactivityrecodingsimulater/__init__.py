def hello() -> str:
    return "Hello from cellactivityrecodingsimulater!"

# 主要モジュールをエクスポート
from . import carsIO
from . import tools

__all__ = [
    'carsIO',
    'tools'
]
