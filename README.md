# Cell Activity Recording Simulator

神経細胞のスパイク信号と記録サイト信号をシミュレートするPythonプロジェクト

## 概要

このプロジェクトは、神経細胞の活動を記録する際の信号をシミュレートし、kilosortのテンプレートファイルを処理・分析するためのツールを提供します。

## インストール

```bash
# 依存関係をインストール
pip install -r requirements.lock

# 開発用依存関係をインストール
pip install -r requirements-dev.lock
```

## 使用方法

### 1. メインプログラムの実行

```bash
# 基本的な実行
python src/cellactivityrecodingsimulater/main.py

# 設定ディレクトリとデータディレクトリを指定
python src/cellactivityrecodingsimulater/main.py --config-dir experiments/ --data-dir data/
```

### 2. Template Maker の使用

#### 設定ファイルモード

```bash
# プロジェクトルートから実行（推奨）
python template_maker_cli.py --create-config my_config.json

# 設定ファイルから実行
python template_maker_cli.py --config my_config.json

# または、srcディレクトリから実行
python src/templateMaker/main.py --create-config my_config.json
python src/templateMaker/main.py --config my_config.json
```

#### 設定ファイルの例

基本的な設定ファイル（`examples/template_maker_config.json`）:

```json
{
  "input": {
    "templates_path": "templates.npy",
    "scale_to_max_one": false
  },
  "filters": {
    "distance": {
      "max_distance": 2.0,
      "method": "euclidean"
    },
    "amplitude": {
      "min_amplitude": 0.1,
      "max_amplitude": 10.0
    },
    "electrode": {
      "min_electrodes": 1,
      "max_electrodes": 10
    },
    "apply_order": ["distance", "amplitude", "electrode"]
  },
  "analysis": {
    "enabled": true,
    "save_analysis": "analysis_results.npz"
  },
  "plot": {
    "enabled": false,
    "max_clusters": 10
  },
  "output": {
    "save_filtered": "filtered_templates.npy",
    "verbose": true
  }
}
```

高度な設定ファイル（`examples/template_maker_config_advanced.json`）:

```json
{
  "input": {
    "templates_path": "templates.npy",
    "scale_to_max_one": true
  },
  "filters": {
    "distance": {
      "max_distance": 1.5,
      "min_distance": 0.1,
      "method": "cosine",
      "reference_cluster": 0
    },
    "amplitude": {
      "min_amplitude": 0.5,
      "max_amplitude": 5.0
    },
    "electrode": {
      "min_electrodes": 2,
      "max_electrodes": 8
    },
    "custom": {
      "conditions": [
        {
          "type": "peak_amplitude_above_mean",
          "multiplier": 1.2
        }
      ]
    },
    "apply_order": ["distance", "amplitude", "electrode", "custom"]
  },
  "analysis": {
    "enabled": true,
    "save_analysis": "advanced_analysis_results.npz"
  },
  "plot": {
    "enabled": true,
    "max_clusters": 15,
    "save_plots": "plots/"
  },
  "output": {
    "save_filtered": "advanced_filtered_templates.npy",
    "save_analysis": "advanced_analysis_results.npz",
    "save_plots": "plots/",
    "verbose": true
  }
}
```

### 3. Python API の使用

```python
from templateMaker import TemplateMaker, run_from_config
from pathlib import Path

# 直接使用（スケーリングなし）
tm = TemplateMaker("templates.npy")
max_amplitudes = tm.get_max_amplitudes()
analysis = tm.analyze_templates()

# スケーリング付きで使用
tm_scaled = TemplateMaker("templates.npy", scale_to_max_one=True)
max_amplitudes_scaled = tm_scaled.get_max_amplitudes()

# フィルタリング
filtered_templates, selected_indices = tm.filter_by_amplitude_range(0.1, 10.0)

# 設定ファイルから実行
run_from_config(Path("my_config.json"))
```

## フィルタリング機能

### 距離フィルタリング
- **ユークリッド距離**: テンプレート間のユークリッド距離
- **コサイン距離**: テンプレート間のコサイン類似度
- **相関距離**: テンプレート間の相関係数

### 振幅フィルタリング
- 最大振幅の範囲でフィルタリング
- ピーク振幅の範囲でフィルタリング

### 電極数フィルタリング
- アクティブな電極数の範囲でフィルタリング

### カスタムフィルタリング
- ユーザー定義の条件でフィルタリング

## スケーリング機能

テンプレートを最大値1にスケーリングする機能を提供します。これは理論的には正しくない可能性がありますが、比較分析や可視化の目的で使用できます。

### 使用方法

#### 設定ファイル
```json
{
  "input": {
    "templates_path": "templates.npy",
    "scale_to_max_one": true
  }
}
```

#### Python API
```python
# スケーリング付きでTemplateMakerを初期化
tm = TemplateMaker("templates.npy", scale_to_max_one=True)
```

## 設定ファイルの詳細

### 入力設定
```json
"input": {
  "templates_path": "templates.npy",  // kilosortテンプレートファイルのパス
  "scale_to_max_one": false           // テンプレートを最大値1にスケーリングするかどうか
}
```

### フィルタリング設定
```json
"filters": {
  "distance": {
    "max_distance": 2.0,           // 最大距離閾値
    "min_distance": null,          // 最小距離閾値
    "method": "euclidean",         // 距離計算方法
    "reference_cluster": null      // 基準クラスタ（指定時はそのクラスタとの距離）
  },
  "amplitude": {
    "min_amplitude": 0.1,          // 最小振幅閾値
    "max_amplitude": 10.0          // 最大振幅閾値
  },
  "electrode": {
    "min_electrodes": 1,           // 最小電極数
    "max_electrodes": 10           // 最大電極数
  },
  "apply_order": ["distance", "amplitude", "electrode"]  // フィルタリング適用順序
}
```

### 分析設定
```json
"analysis": {
  "enabled": true,                 // 分析を実行するかどうか
  "save_analysis": "results.npz"   // 分析結果の保存先
}
```

### プロット設定
```json
"plot": {
  "enabled": false,                // プロットを実行するかどうか
  "max_clusters": 10,              // プロットする最大クラスタ数
  "save_plots": null               // プロットの保存先ディレクトリ
}
```

### 出力設定
```json
"output": {
  "save_filtered": "filtered.npy", // フィルタリングされたテンプレートの保存先
  "save_analysis": "results.npz",  // 分析結果の保存先
  "save_plots": null,              // プロットの保存先ディレクトリ
  "verbose": true                  // 詳細な出力を表示するかどうか
}
```

## テスト

```bash
# 設定ファイルランナーのテスト
python test_config_runner.py

# kilosortテンプレートのテスト
python test_kilosort_templates.py
```

## プロジェクト構造

```
CellActivityRecodingSimulater/
├── src/
│   ├── cellactivityrecodingsimulater/     # メインプロジェクト
│   │   ├── main.py                        # メインプログラム
│   │   ├── Cell.py                        # 細胞クラス
│   │   ├── Site.py                        # 記録サイトクラス
│   │   ├── Settings.py                    # 設定クラス
│   │   ├── carsIO.py                      # I/O機能
│   │   └── tools.py                       # ユーティリティ
│   └── templateMaker/                     # テンプレート処理パッケージ
│       ├── __init__.py
│       ├── main.py                        # CLIエントリーポイント
│       ├── template_analyzer.py           # テンプレート解析クラス
│       ├── config_schema.py               # 設定ファイルスキーマ
│       └── config_runner.py               # 設定ファイルランナー
├── examples/                              # 設定ファイル例
│   ├── template_maker_config.json
│   └── template_maker_config_advanced.json
├── tests/                                 # テストデータ
│   └── data/
├── experiments/                           # 実験条件ファイル
├── template_maker_cli.py                  # CLIエントリーポイント（プロジェクトルート）
├── pyproject.toml                        # プロジェクト設定
├── requirements.lock                     # 依存関係
└── README.md                             # このファイル
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。


