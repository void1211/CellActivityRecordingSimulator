#!/usr/bin/env python3
"""
Cell Activity Recording Simulator - エントリーポイント
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# メインプログラムをインポートして実行
from cellactivityrecodingsimulater.main import main

if __name__ == "__main__":
    main() 
