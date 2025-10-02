# Cell Activity Recording Simulator

神経細胞のスパイク信号と記録サイト信号をシミュレートするPythonプロジェクト

## 概要

このプロジェクトは、神経細胞の活動を記録する際の信号をシミュレートするためのツールを提供します。

## インストール

```bash
# 依存関係をインストール
pip install -r requirements.lock

# 開発用依存関係をインストール
pip install -r requirements-dev.lock
```

## 使用方法

### メインプログラムの実行

```bash
# 基本的な実行
python main.py --example-dir 実験条件/ex-1

# 特定の条件フォルダパターンを指定
python main.py --example-dir 実験条件/ex-1 --conditions "condition*"

# プロット表示を無効化
python main.py --example-dir 実験条件/ex-1 --no-plot

# 環境変数を使用
export EXAMPLE_DIR="実験条件/ex-1"
python main.py
```

### 設定ファイルの例

条件フォルダ内の設定ファイル（`実験条件/ex-1/condition1/settings.json`）:

```json
{
    "name": "2025_ex-1_condition1",
    "pathCell": "cells.json",
    "pathSite": "sites.json",
    "pathSaveDir": null,
    "fs": 30000,
    "duration": 10,
    "avgSpikeRate": 10,
    "isRefractory": true,
    "refractoryPeriod": 10,
    "noiseType": "gaussian",
    "noiseAmp": 10,
    "spikeType": "gabor",
    "spikeAmpMax": 100,
    "spikeAmpMin": 90,
    "attenTime": 25
}
```

## プロジェクト構造

```
cellactivityrecodingsimulator/
├── src/
│   └── cellactivityrecodingsimulator/     # メインプロジェクト
│       ├── main.py                        # メインプログラム
│       ├── Cell.py                        # 細胞クラス
│       ├── Site.py                        # 記録サイトクラス
│       ├── Settings.py                    # 設定クラス
│       ├── carsIO.py                      # I/O機能
│       └── tools.py                       # ユーティリティ
├── 実験条件/                              # 実験条件ファイル
│   └── ex-1/
│       ├── condition1/                    # 条件1
│       │   ├── settings.json             # 設定ファイル
│       │   ├── cells.json                # セルデータ
│       │   ├── sites.json                # プローブデータ
│       │   └── results/                  # 結果保存先
│       └── condition2/                    # 条件2
│           ├── settings.json             # 設定ファイル
│           ├── cells.json                # セルデータ
│           ├── sites.json                # プローブデータ
│           └── results/                  # 結果保存先
├── main.py                               # エントリーポイント
├── pyproject.toml                        # プロジェクト設定
├── requirements.lock                     # 依存関係
└── README.md                             # このファイル
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。


