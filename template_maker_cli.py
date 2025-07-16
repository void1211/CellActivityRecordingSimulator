#!/usr/bin/env python3
"""
Template Maker CLI Entry Point

プロジェクトルートから実行するためのCLIエントリーポイント
設定ファイルモードのみ対応
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# templateMakerのmain.pyを実行
if __name__ == "__main__":
    from templateMaker.main import main
    main() 
