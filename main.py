#!/usr/bin/env python3
"""
Cell Activity Recording Simulator - エントリーポイント
"""

import sys
import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Cell Activity Recording Simulator')
    parser.add_argument(
        '--example-dir', '-ed',
        type=str,
        default=os.environ.get('EXAMPLE_DIR'),
        help='Path to example directory. Can also be set via EXAMPLE_DIR env var.'
    )
    parser.add_argument(
        '--conditions', '-c',
        type=str,
        default='condition*',
        help='Condition folder pattern (default: condition*)'
    )
    parser.add_argument(
        '--no-plot', '-np',
        action='store_true',
        help='Disable plot display'
    )

    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode'
    )

    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Debug mode'
    )
    
    args = parser.parse_args()
    
    # 必須パラメータのチェック
    if not args.example_dir:
        parser.error("--example-dir is required or set EXAMPLE_DIR environment variable")
    
    return args

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "cellactivityrecodingsimulater"))
sys.path.insert(1, str(Path("C:/Users/tanaka-users/tlab/tlab_yasui/2025/simulations")))

# メインプログラムをインポートして実行
from cellactivityrecodingsimulater.main import main

if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).parent
    main(project_root=project_root, example_dir=args.example_dir, condition_pattern=args.conditions, show_plot=not args.no_plot, test=args.test, debug=args.debug) 
