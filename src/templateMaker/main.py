#!/usr/bin/env python3
"""
Template Maker Main Entry Point

kilosortテンプレートファイルを処理するためのメインプログラム
"""

import sys
import argparse
from pathlib import Path

# 相対インポートを絶対インポートに変更
try:
    from templateMaker.config_runner import run_from_config
    from templateMaker.config_schema import create_default_config, save_config
except ImportError:
    # 直接実行時のためのフォールバック
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from templateMaker.config_runner import run_from_config
    from templateMaker.config_schema import create_default_config, save_config


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='kilosortテンプレートファイルを処理します')
    
    # 実行モードの選択
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--config', type=str, help='設定ファイルのパス')
    mode_group.add_argument('--create-config', type=str, help='デフォルト設定ファイルを作成')
    
    args = parser.parse_args()
    
    # 設定ファイル作成モード
    if args.create_config:
        config_path = Path(args.create_config)
        config = create_default_config()
        save_config(config, config_path)
        print(f"デフォルト設定ファイルを作成しました: {config_path}")
        return
    
    # 設定ファイル実行モード
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"エラー: 設定ファイルが見つかりません: {config_path}")
            sys.exit(1)
        
        print(f"設定ファイルから実行します: {config_path}")
        run_from_config(config_path)
        return


if __name__ == "__main__":
    main()
