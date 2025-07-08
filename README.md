# Cell Activity Recording Simulator

神経細胞活動記録シミュレーター - 複数の神経細胞からのスパイク信号をシミュレートし、記録サイトでの信号を生成するPythonパッケージです。

## 概要

このプロジェクトは、神経科学研究における細胞外記録のシミュレーションを行うためのツールです。複数の神経細胞のスパイク活動を模擬し、記録電極での信号を生成します。

### 主な機能

- **スパイク時間のシミュレーション**: ポアソン過程に基づくスパイク生成
- **スパイクテンプレート**: ガボール関数によるスパイク波形生成
- **ノイズシミュレーション**: ガウシアンノイズまたは実測ノイズの適用
- **距離減衰**: セルと記録サイト間の距離による振幅減衰
- **複数セル・複数サイト**: 複数の神経細胞と記録電極の同時シミュレーション

## インストール

### 前提条件

- Python 3.10以上
- pip または rye

### インストール方法

1. リポジトリをクローン
```bash
git clone <repository-url>
cd CellActivityRecodingSimulater
```

2. 依存関係のインストール
```bash
# pipを使用する場合
pip install -r requirements.lock

# ryeを使用する場合
rye sync
```

## 使用方法

### 基本的な実行

```bash
# 仮想環境を有効化
source .venv/bin/activate

# メインプログラムを実行
python src/cellactivityrecodingsimulater/main.py
```

### 設定ファイル

`tests/data/test_settings.json`でシミュレーション条件を設定できます：

```json
{
    "pathCell": "test_cells.json",
    "pathSite": "test_sites.json",
    "pathSaveDir": "test_save",
    "fs": 1000,
    "duration": 10,
    "avgSpikeRate": 10,
    "isRefractory": true,
    "refractoryPeriod": 10,
    "noiseType": "gaussian",
    "noiseAmp": 10,
    "spikeType": "gabor",
    "isRandomSelect": true,
    "gaborSigmaList": [1, 2, 3],
    "gaborf0List": [1, 2, 3],
    "gaborthetaList": [1, 2, 3],
    "spikeWidth": 4,
    "spikeAmp": 10,
    "attenTime": 10
}
```

### 設定パラメータ

| パラメータ | 説明 | 単位 |
|-----------|------|------|
| `fs` | サンプリング周波数 | Hz |
| `duration` | シミュレーション時間 | 秒 |
| `avgSpikeRate` | 平均スパイク発火率 | Hz |
| `noiseType` | ノイズタイプ | "gaussian", "truth", "none" |
| `spikeType` | スパイクタイプ | "gabor", "templates" |
| `spikeWidth` | スパイク幅 | ミリ秒 |

## プロジェクト構造

```
CellActivityRecodingSimulater/
├── src/
│   └── cellactivityrecodingsimulater/
│       ├── main.py          # メインプログラム
│       ├── Cell.py          # 神経細胞クラス
│       ├── Site.py          # 記録サイトクラス
│       ├── Settings.py      # 設定クラス
│       ├── tools.py         # シミュレーション関数
│       ├── carsIO.py        # 入出力関数
│       └── __init__.py
├── tests/
│   └── data/                # テストデータ
│       ├── test_settings.json
│       ├── test_cells.json
│       ├── test_sites.json
│       └── test_save/
├── pyproject.toml
├── requirements.lock
└── README.md
```

## 主要クラス

### Cell
神経細胞を表現するクラス
- `id`: セルID
- `x, y, z`: 3次元座標
- `spikeTimeList`: スパイク発生時刻リスト
- `spikeAmpList`: スパイク振幅リスト
- `spikeTemp`: スパイクテンプレート

### Site
記録サイトを表現するクラス
- `id`: サイトID
- `x, y, z`: 3次元座標
- `signalRaw`: 生信号
- `signalNoise`: ノイズ信号

### Settings
シミュレーション設定を管理するクラス
- サンプリング周波数、時間、ノイズ設定など

## 開発

### 依存関係

- **pydantic**: データバリデーション
- **numpy**: 数値計算
- **matplotlib**: 可視化

### 開発環境のセットアップ

```bash
# 開発用依存関係のインストール
rye sync --dev

# テストの実行
python -m pytest tests/
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 作者

- void1211 (g2154488@cc.kyoto-su.ac.jp)

## 謝辞

このプロジェクトは神経科学研究のためのシミュレーションツールとして開発されました。


