# SAM3 APIリファレンス

## 概要

**SAM3 (Segment Anything Model 3)** は、Meta社が開発した画像・動画セグメンテーションのための統合基盤モデルです。テキストプロンプト、ポイント、バウンディングボックス、マスクなど、様々なプロンプトを使用してオブジェクトの検出、セグメンテーション、トラッキングを実行できます。

### 主な特徴

- **オープンボキャブラリーセグメンテーション**: SA-COベンチマークで270,000以上のユニークな概念に対応
- **マルチモーダルプロンプト**: テキスト、ポイント、ボックス、マスクプロンプトをサポート
- **動画トラッキング**: SAM2のTransformerアーキテクチャを継承し、時間的一貫性を実現
- **プレゼンストークンアーキテクチャ**: 類似プロンプトの区別（例：異なる色のオブジェクト）
- **デカップルド検出器-トラッカー設計**: タスク間の干渉を最小化し、効率的にスケール

### システム要件

- Python 3.12以上
- PyTorch 2.7以上
- CUDA 12.6以上対応GPU
- Hugging Faceアカウント（モデルチェックポイントのアクセス用）

---

## 目次

1. [インストール](#1-インストール)
2. [主要モジュール構造](#2-主要モジュール構造)
3. [画像セグメンテーションAPI](#3-画像セグメンテーションapi)
4. [動画セグメンテーション・トラッキングAPI](#4-動画セグメンテーショントラッキングapi)
5. [Transformerアーキテクチャ](#5-transformerアーキテクチャ)
6. [バックボーンと特徴抽出](#6-バックボーンと特徴抽出)
7. [エンコーダー（ジオメトリとテキスト）](#7-エンコーダージオメトリとテキスト)
8. [セグメンテーションヘッドとユーティリティ](#8-セグメンテーションヘッドとユーティリティ)
9. [モデルビルダー関数](#9-モデルビルダー関数)
10. [エージェントインターフェース](#10-エージェントインターフェース)
11. [ユーティリティ関数](#11-ユーティリティ関数)
12. [評価モジュール](#12-評価モジュール)
13. [使用例](#13-使用例)

---

## 1. インストール

### 基本インストール

```bash
# Conda環境の作成
conda create -n sam3 python=3.12 -y
conda activate sam3

# PyTorchのインストール（CUDA 12.6サポート）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# リポジトリのクローンとインストール
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# ノートブック用の追加依存関係（オプション）
pip install -e ".[notebooks]"

# 開発用依存関係（オプション）
pip install -e ".[dev]"
```

### モデルチェックポイントのダウンロード

```python
from sam3.model_builder import download_ckpt_from_hf

# Hugging Faceから自動ダウンロード
checkpoint_path = download_ckpt_from_hf()
```

---

## 2. 主要モジュール構造

```
sam3/
├── __init__.py                 # メインパッケージエントリポイント
├── logger.py                   # カラーフォーマット付きロギング
├── model_builder.py            # モデル構築ファクトリ関数
├── visualization_utils.py      # 可視化とレンダリングユーティリティ
│
├── model/                      # コアモデルアーキテクチャ
│   ├── sam3_image.py          # 画像セグメンテーションモデル
│   ├── sam3_video_base.py     # 動画処理とトラッキングのベース
│   ├── sam3_video_inference.py # 動画推論パイプライン
│   ├── sam3_video_predictor.py # 動画予測インターフェース
│   ├── sam3_tracking_predictor.py # インタラクティブ動画トラッキング
│   ├── encoder.py             # Transformerエンコーダーコンポーネント
│   ├── decoder.py             # Transformerデコーダーコンポーネント
│   ├── memory.py              # トラッキング用メモリ管理
│   ├── geometry_encoders.py   # ジオメトリプロンプトエンコーディング
│   ├── text_encoder_ve.py     # Vision-Languageテキストエンコーディング
│   ├── vitdet.py              # Vision Transformerバックボーン
│   ├── vl_combiner.py         # Vision-Languageバックボーンラッパー
│   ├── necks.py               # 特徴ピラミッドネック
│   ├── position_encoding.py   # 正弦波位置埋め込み
│   ├── maskformer_segmentation.py # セグメンテーションヘッド
│   ├── sam3_image_processor.py   # 画像前処理
│   ├── box_ops.py             # ボックス操作ユーティリティ
│   ├── io_utils.py            # I/Oとリソース読み込み
│   ├── data_misc.py           # データ構造とユーティリティ
│   ├── model_misc.py          # モデルユーティリティクラス
│   └── edt.py                 # ユークリッド距離変換
│
├── agent/                      # マルチモーダルエージェント統合
│   ├── agent_core.py          # エージェント推論オーケストレーション
│   ├── client_sam3.py         # SAM3サービスクライアント
│   ├── client_llm.py          # LLMクライアントインターフェース
│   └── helpers/               # ヘルパーユーティリティ
│       ├── boxes.py           # ボックスフォーマット変換
│       ├── masks.py           # マスク処理
│       ├── rle.py             # RLEエンコード/デコード
│       ├── color_map.py       # カラーマッピング
│       ├── keypoints.py       # キーポイント処理
│       ├── visualizer.py      # 可視化レンダリング
│       ├── mask_overlap_removal.py # オーバーラップ除去
│       ├── roi_align.py       # ROIアラインメント
│       └── som_utils.py       # Self-organizing mapユーティリティ
│
└── eval/                       # 評価とベンチマーク
    ├── coco_eval.py           # COCOベンチマーク評価
    ├── ytvis_eval.py          # YouTube-VIS評価
    └── saco_eval.py           # SAM3データセット評価
```

---

## 3. 画像セグメンテーションAPI

### 3.1 Sam3Image

`sam3/model/sam3_image.py`

Vision-Language機能を備えた画像セグメンテーション用のメインPyTorchモジュール。

#### コンストラクタパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|----------|------|---------|------|
| `backbone` | SAM3VLBackbone | 必須 | 特徴抽出バックボーン |
| `transformer` | nn.Module | 必須 | Transformerアーキテクチャ |
| `input_geometry_encoder` | nn.Module | 必須 | ジオメトリプロンプトエンコーダー |
| `segmentation_head` | nn.Module | None | マスク予測ヘッド |
| `num_feature_levels` | int | 4 | 特徴ピラミッドレベル数 |
| `o2m_mask_predict` | bool | False | 1対多マスク予測 |
| `use_instance_query` | bool | False | インスタンスクエリ使用 |
| `multimask_output` | bool | False | 複数マスク出力 |
| `use_dot_prod_scoring` | bool | False | ドット積分類 |
| `supervise_joint_box_scores` | bool | False | ジョイントボックス教師あり学習 |
| `separate_scorer_for_instance` | bool | False | インスタンス別スコアリング |
| `num_interactive_steps_val` | int | 5 | インタラクティブ改善ステップ数 |

#### 主要メソッド

```python
def forward(
    image_inputs: Tensor,
    text_inputs: List[str],
    geometric_inputs: Dict
) -> Dict:
    """
    フォワードパス - 画像、テキスト、ジオメトリプロンプトを処理

    Args:
        image_inputs: 入力画像テンソル [B, C, H, W]
        text_inputs: テキストプロンプトのリスト
        geometric_inputs: {boxes, points, masks}を含む辞書

    Returns:
        {masks, boxes, scores, labels}を含む辞書
    """

def forward_grounding(
    image: Tensor,
    text_prompt: str,
    boxes: Optional[Tensor] = None,
    points: Optional[Tensor] = None
) -> Dict:
    """
    グラウンディング推論 - テキストとオプションのジオメトリプロンプト

    Args:
        image: 入力画像 [C, H, W]
        text_prompt: テキスト記述
        boxes: オプションのバウンディングボックス [N, 4]
        points: オプションのポイント [N, 2]

    Returns:
        セグメンテーション出力
    """

def predict_inst(image: Tensor) -> Dict:
    """
    インスタンス予測 - 単一画像

    Returns:
        {masks, boxes, scores, labels}
    """

def predict_inst_batch(images: List[Tensor]) -> List[Dict]:
    """
    バッチインスタンス予測

    Returns:
        予測のリスト
    """
```

### 3.2 Sam3Processor

`sam3/model/sam3_image_processor.py`

画像の前処理と推論を処理する高レベルインターフェース。

#### コンストラクタパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|----------|------|---------|------|
| `model` | Sam3Image | 必須 | SAM3モデルインスタンス |
| `resolution` | int | 1008 | 処理解像度 |
| `device` | str | "cuda" | 計算デバイス |
| `confidence_threshold` | float | 0.5 | 検出閾値 |

#### 主要メソッド

```python
def set_image(image: Union[PIL.Image, Tensor]) -> None:
    """
    処理する画像を設定

    Args:
        image: PIL画像またはテンソル
    """

def set_image_batch(images: List) -> None:
    """
    バッチ処理用の複数画像を設定

    Args:
        images: 画像のリスト
    """

def set_text_prompt(text: str) -> None:
    """
    テキストプロンプトを設定

    Args:
        text: テキスト記述（例: "cat", "red car"）
    """

def add_geometric_prompt(
    bbox: Optional[Tensor] = None,
    label: Optional[int] = None,
    mask: Optional[Tensor] = None
) -> None:
    """
    ジオメトリプロンプトを追加

    Args:
        bbox: バウンディングボックス [x0, y0, x1, y1]
        label: クラスラベル
        mask: バイナリマスクテンソル
    """

def reset_all_prompts() -> None:
    """すべてのプロンプトをクリア"""

def set_confidence_threshold(threshold: float) -> None:
    """
    信頼度閾値を設定

    Args:
        threshold: 0.0～1.0の閾値
    """
```

#### 出力フォーマット

```python
{
    'masks': Tensor,        # [N, H, W] ブールテンソル
    'boxes': Tensor,        # [N, 4] [x0, y0, x1, y1]フォーマット
    'scores': Tensor,       # [N] 信頼度スコア
    'labels': Tensor,       # [N] クラスラベル
}
# すべての出力は元の画像サイズにリスケールされます
```

---

## 4. 動画セグメンテーション・トラッキングAPI

### 4.1 Sam3VideoInference

`sam3/model/sam3_video_inference.py`

検出とトラッキングをサポートする動画推論オーケストレーション。

#### 主要メソッド

```python
def init_state(image_or_video_path: str) -> None:
    """
    動画またはフレームシーケンスで状態を初期化

    Args:
        image_or_video_path: 動画ファイルまたは画像ディレクトリへのパス
    """

def reset_state() -> None:
    """推論状態をリセット"""

def add_prompt(
    frame_idx: int,
    prompt_type: str,
    prompt_data: Any
) -> None:
    """
    特定フレームにプロンプトを追加

    Args:
        frame_idx: フレームインデックス
        prompt_type: 'text', 'box', 'point', 'mask'
        prompt_data: プロンプトデータ
    """

def propagate_in_video() -> Generator[Dict]:
    """
    動画全体でセグメンテーションを伝播

    Yields:
        フレーム出力辞書
    """

def warm_up_compilation() -> None:
    """モデルコンパイルのウォームアップ"""
```

### 4.2 Sam3VideoInferenceWithInstanceInteractivity

`Sam3VideoInference`を拡張し、ポイントベースのオブジェクト改善機能を追加。

#### 追加メソッド

```python
def add_tracker_new_points(
    object_id: int,
    frame_idx: int,
    points: Tensor
) -> None:
    """
    トラッキングオブジェクトに改善ポイントを追加

    Args:
        object_id: オブジェクトID
        frame_idx: フレームインデックス
        points: ポイント座標 [N, 2]
    """

def remove_object(object_id: int) -> None:
    """
    トラッキングからオブジェクトを削除

    Args:
        object_id: 削除するオブジェクトID
    """

def add_action_history(
    action_type: str,
    object_id: int,
    frame_idx: int
) -> None:
    """
    アクション履歴を記録

    Args:
        action_type: アクションタイプ
        object_id: オブジェクトID
        frame_idx: フレームインデックス
    """
```

### 4.3 Sam3TrackerPredictor

`sam3/model/sam3_tracking_predictor.py`

ユーザーインタラクションとマルチオブジェクトサポートを備えたインタラクティブ動画オブジェクトトラッキング。

#### 主要メソッド

```python
def init_state(
    video_path: str,
    video_info: Dict
) -> None:
    """
    動画でトラッカーを初期化

    Args:
        video_path: 動画ファイルパス
        video_info: 動画メタデータ
    """

def add_new_points_or_box(
    frame_idx: int,
    points: Optional[Tensor] = None,
    box: Optional[Tensor] = None
) -> None:
    """
    新しいオブジェクトまたは既存オブジェクトの改善を追加

    Args:
        frame_idx: フレームインデックス
        points: ポイント座標 [N, 2]（正 or 負）
        box: バウンディングボックス [x0, y0, x1, y1]
    """

def add_new_mask(
    frame_idx: int,
    mask_tensor: Tensor
) -> None:
    """
    マスクプロンプトを追加

    Args:
        frame_idx: フレームインデックス
        mask_tensor: バイナリマスク [H, W]
    """

def clear_all_points_in_frame(frame_idx: int) -> None:
    """特定フレームのすべてのポイントをクリア"""

def clear_all_points_in_video() -> None:
    """動画全体のすべてのポイントをクリア"""

def propagate_in_video(
    forward: bool = True,
    backward: bool = True
) -> None:
    """
    動画でマスクを伝播

    Args:
        forward: 前方向に伝播
        backward: 後方向に伝播
    """

def remove_object(object_id: int) -> None:
    """オブジェクトを削除"""
```

### 4.4 Sam3VideoPredictor

`sam3/model/sam3_video_predictor.py`

動画推論セッションを管理。

#### 主要メソッド

```python
def start_session(image_or_video_path: str) -> str:
    """
    新しいセッションを開始

    Args:
        image_or_video_path: 動画パス

    Returns:
        session_id: セッション識別子
    """

def close_session(session_id: str) -> None:
    """セッションを閉じる"""

def reset_session(session_id: str) -> None:
    """セッションをリセット"""

def add_prompt(
    session_id: str,
    frame_idx: int,
    prompt_type: str,
    prompt_data: Any
) -> None:
    """プロンプトを追加"""

def propagate_in_video(
    session_id: str,
    direction: str = "both"
) -> Generator[Dict]:
    """
    動画で伝播

    Args:
        session_id: セッションID
        direction: "forward", "backward", または "both"

    Yields:
        フレーム出力
    """

def handle_request(request: Dict) -> Dict:
    """
    同期リクエストを処理

    Args:
        request: リクエスト辞書

    Returns:
        レスポンス辞書
    """

def handle_stream_request(request: Dict) -> Iterator[Dict]:
    """
    ストリーミングリクエストを処理

    Yields:
        レスポンスチャンク
    """
```

---

## 5. Transformerアーキテクチャ

### 5.1 TransformerEncoderLayer

`sam3/model/encoder.py`

#### コンストラクタパラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `activation` | str | 活性化関数タイプ |
| `cross_attention` | nn.Module | クロスアテンションモジュール |
| `d_model` | int | モデル次元 |
| `dim_feedforward` | int | FFN隠れ次元 |
| `dropout` | float | ドロップアウト率 |
| `pos_enc_at_attn` | bool | アテンションでの位置エンコーディング |
| `pre_norm` | bool | Pre-normalizationバリアント |
| `self_attention` | nn.Module | セルフアテンションモジュール |

#### メソッド

```python
def forward(
    tgt: Tensor,
    memory: Tensor,
    dac: bool = False,
    **kwargs
) -> Tensor:
    """
    エンコーダーレイヤーのフォワードパス

    Args:
        tgt: ターゲットテンソル
        memory: メモリテンソル
        dac: Deformable attention control

    Returns:
        エンコードされた特徴
    """
```

### 5.2 TransformerEncoder

`sam3/model/encoder.py`

#### コンストラクタパラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `layer` | nn.Module | エンコーダーレイヤー |
| `num_layers` | int | レイヤー数 |
| `d_model` | int | モデル次元 |
| `num_feature_levels` | int | 特徴ピラミッドレベル |
| `frozen` | bool | パラメータを凍結 |
| `use_act_checkpoint` | bool | 活性化チェックポイント |

#### メソッド

```python
def forward(
    src: List[Tensor],
    src_key_padding_masks: Optional[Tensor] = None,
    pos: Optional[Tensor] = None,
    prompt: Optional[Tensor] = None,
    prompt_key_padding_mask: Optional[Tensor] = None,
    encoder_extra_kwargs: Optional[Dict] = None
) -> Tuple:
    """
    エンコーダーフォワードパス

    Args:
        src: マルチレベル特徴のリスト
        src_key_padding_masks: パディングマスク
        pos: 位置エンコーディング
        prompt: プロンプト埋め込み

    Returns:
        (memory, prompt_memory, spatial_shapes, level_start_index, valid_ratios, reference_points)
    """

@staticmethod
def get_reference_points(
    spatial_shapes: Tensor,
    valid_ratios: Tensor,
    device: torch.device
) -> Tensor:
    """参照ポイントを生成"""
```

### 5.3 TransformerEncoderFusion

Vision-Language融合機能を持つ`TransformerEncoder`の拡張。

#### 追加パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|----------|------|---------|------|
| `add_pooled_text_to_img_feat` | bool | True | プールされたテキスト特徴を追加 |
| `pool_text_with_mask` | bool | False | マスク付きプーリング |
| `compile_mode` | str | None | コンパイルモード |

### 5.4 TransformerDecoder

`sam3/model/decoder.py`

#### 主要メソッド

```python
def forward(
    tgt: Tensor,
    memory: Tensor,
    tgt_mask: Optional[Tensor] = None,
    memory_mask: Optional[Tensor] = None,
    tgt_key_padding_mask: Optional[Tensor] = None,
    memory_key_padding_mask: Optional[Tensor] = None,
    pos: Optional[Tensor] = None,
    query_pos: Optional[Tensor] = None,
    reference_points: Optional[Tensor] = None,
    spatial_shapes: Optional[Tensor] = None,
    valid_ratios: Optional[Tensor] = None
) -> Dict:
    """
    デコーダーフォワードパス

    Returns:
        デコーダー出力辞書
    """
```

---

## 6. バックボーンと特徴抽出

### 6.1 ViT (Vision Transformer)

`sam3/model/vitdet.py`

#### コアコンポーネント

- **PatchEmbed**: 画像をパッチ埋め込みに変換
- **Attention**: RoPEサポート付きマルチヘッドセルフアテンション
- **Block**: アテンション + FFNを持つTransformerブロック
- **ViT**: 完全なバックボーン

#### ViT コンストラクタパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|----------|------|---------|------|
| `img_size` | int | 1024 | 入力画像サイズ |
| `patch_size` | int | 16 | パッチサイズ |
| `in_chans` | int | 3 | 入力チャンネル数 |
| `embed_dim` | int | 768 | 埋め込み次元 |
| `depth` | int | 12 | レイヤー数 |
| `num_heads` | int | 12 | アテンションヘッド数 |
| `mlp_ratio` | float | 4.0 | MLP次元比 |
| `qkv_bias` | bool | True | QKV投影バイアス |
| `drop_rate` | float | 0.0 | ドロップアウト率 |
| `attn_drop_rate` | float | 0.0 | アテンションドロップアウト |
| `drop_path_rate` | float | 0.0 | 確率的深さ |
| `use_abs_pos` | bool | True | 絶対位置埋め込み |
| `use_rel_pos` | bool | True | 相対位置埋め込み |
| `window_size` | int | 0 | ウィンドウアテンションサイズ |
| `use_rope` | bool | False | 回転位置エンコーディング |

### 6.2 SAM3VLBackbone

`sam3/model/vl_combiner.py`

視覚とテキスト特徴を組み合わせるVision-Languageバックボーン。

#### コンストラクタパラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `vision_backbone` | Sam3DualViTDetNeck | 視覚バックボーン |
| `text_backbone` | VETextEncoder | テキストエンコーダー |

#### メソッド

```python
def forward(images: Tensor, captions: List[str]) -> Dict:
    """
    Vision-Languageフォワードパス

    Returns:
        {visual_features, text_features, fusion_output}
    """

def forward_image(images: Tensor) -> Dict:
    """画像のみの処理"""

def forward_text(captions: List[str]) -> Tuple:
    """テキストのみの処理"""
```

### 6.3 Sam3DualViTDetNeck

`sam3/model/necks.py`

特徴ピラミッド生成のためのSimpleFPNスタイルネック。

#### コンストラクタパラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `backbone` | nn.Module | 視覚バックボーン |
| `position_encoding` | nn.Module | 位置エンコーダー |
| `d_model` | int | モデル次元 |
| `add_sam2_neck` | bool | デュアルネックモード |

#### スケールファクター

- 4.0, 2.0, 1.0, 0.5

#### メソッド

```python
def forward(x: Tensor) -> List[Tensor]:
    """
    マルチレベル特徴を生成

    Returns:
        特徴ピラミッドのリスト
    """
```

---

## 7. エンコーダー（ジオメトリとテキスト）

### 7.1 SequenceGeometryEncoder

`sam3/model/geometry_encoders.py`

ジオメトリプロンプト（ボックス、ポイント、マスク）をエンコード。

#### 主要メソッド

```python
def _encode_points(
    points: Tensor,
    img_feats: Tensor
) -> Tensor:
    """ポイントプロンプトをエンコード"""

def _encode_boxes(
    boxes: Tensor,
    img_feats: Tensor
) -> Tensor:
    """ボックスプロンプトをエンコード"""

def _encode_masks(
    masks: Tensor,
    img_feats: Tensor
) -> Tensor:
    """マスクプロンプトをエンコード"""

def forward(
    geo_prompt: Dict,
    img_feats: Tensor,
    img_sizes: Tensor,
    img_pos_embeds: Tensor
) -> Dict:
    """
    ジオメトリエンコーディング

    Args:
        geo_prompt: {boxes, points, masks}
        img_feats: 画像特徴
        img_sizes: 画像サイズ
        img_pos_embeds: 位置埋め込み

    Returns:
        エンコードされたジオメトリ
    """
```

### 7.2 TextTransformer

`sam3/model/text_encoder_ve.py`

Transformerベースのテキストエンコーディング。

#### コンストラクタパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|----------|------|---------|------|
| `context_length` | int | 77 | 最大シーケンス長 |
| `vocab_size` | int | 49408 | 語彙サイズ |
| `width` | int | 512 | 隠れ次元 |
| `heads` | int | 8 | アテンションヘッド数 |
| `layers` | int | 12 | レイヤー数 |
| `mlp_ratio` | float | 4.0 | MLP比 |
| `output_dim` | int | 512 | 出力次元 |
| `causal_mask` | bool | True | 因果マスク使用 |
| `pool_type` | str | "argmax" | プーリング戦略 |

#### メソッド

```python
def forward(x: Tensor) -> Tensor:
    """
    テキストTransformerフォワードパス

    Args:
        x: トークンID [B, L]

    Returns:
        エンコードされたテキスト特徴 [B, D]
    """
```

### 7.3 VETextEncoder

`sam3/model/text_encoder_ve.py`

トークン化と埋め込みリサイズを統合したラッパー。

#### コンストラクタパラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `d_model` | int | モデル次元 |
| `tokenizer` | Callable | トークン化関数 |

#### メソッド

```python
def forward(x: Union[List[str], Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    """
    テキストエンコーダーフォワードパス

    Args:
        x: テキストリストまたはトークンテンソル

    Returns:
        (attention_masks, transformed_memory, embedded_inputs)
    """
```

### 7.4 PositionEmbeddingSine

`sam3/model/position_encoding.py`

"Attention is All You Need"からの正弦波位置埋め込み。

#### メソッド

```python
def forward(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """位置埋め込みを生成"""

def encode_boxes(boxes: Tensor) -> Tensor:
    """ボックス座標をエンコード"""

def encode_points(points: Tensor, labels: Tensor) -> Tensor:
    """ポイント座標とラベルをエンコード"""
```

---

## 8. セグメンテーションヘッドとユーティリティ

### 8.1 UniversalSegmentationHead

`sam3/model/maskformer_segmentation.py`

セマンティック + インスタンスセグメンテーションヘッド。

#### コンストラクタパラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `hidden_dim` | int | 隠れ次元 |
| `upsampling_stages` | int | アップサンプリングレイヤー数 |
| `use_encoder_inputs` | bool | エンコーダー入力使用 |
| `aux_masks` | bool | 補助マスク出力 |
| `no_dec` | bool | デコーダーをスキップ |

#### コアコンポーネント

- **presence_head**: オブジェクト検出スコアリング
- **cross_attention**: プロンプト統合
- セマンティックとインスタンスセグメンテーション用の個別ヘッド

#### メソッド

```python
def forward(features: Tensor, prompts: Tensor) -> Dict:
    """
    セグメンテーションヘッドフォワードパス

    Returns:
        {masks, semantic_segmentation, presence_logits}
    """
```

---

## 9. モデルビルダー関数

`sam3/model_builder.py`

### 9.1 主要ビルダー関数

```python
def build_sam3_image_model(
    bpe_path: Optional[str] = None,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    compile: bool = False,
    eval_mode: bool = True,
    enable_inst_interactivity: bool = False
) -> Sam3Image:
    """
    SAM3画像モデルを構築

    Args:
        bpe_path: BPEトークナイザーパス
        device: 計算デバイス
        checkpoint_path: モデルチェックポイント
        compile: Torchコンパイル有効化
        eval_mode: 評価モード
        enable_inst_interactivity: インスタンスインタラクティビティ

    Returns:
        Sam3Imageモデル
    """

def build_tracker(
    apply_temporal_disambiguation: bool,
    with_backbone: bool = False,
    compile_mode: Optional[str] = None
) -> Sam3TrackerPredictor:
    """
    動画トラッカーを構築

    Args:
        apply_temporal_disambiguation: 時間的曖昧性除去適用
        with_backbone: バックボーン含む
        compile_mode: コンパイルモード

    Returns:
        Sam3TrackerPredictorインスタンス
    """

def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    bpe_path: Optional[str] = None,
    device: str = "cuda",
    compile: bool = False
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    SAM3動画モデルを構築

    Args:
        checkpoint_path: モデルチェックポイント
        bpe_path: BPEトークナイザーパス
        device: 計算デバイス
        compile: コンパイル有効化

    Returns:
        動画推論モデル
    """

def build_sam3_video_predictor(
    *model_args,
    gpus_to_use: Optional[List[int]] = None,
    **model_kwargs
) -> Sam3VideoPredictor:
    """
    動画プレディクタを構築

    Args:
        gpus_to_use: 使用するGPUのリスト

    Returns:
        Sam3VideoPredictor
    """

def download_ckpt_from_hf() -> str:
    """
    Hugging Faceからチェックポイントをダウンロード

    Returns:
        チェックポイントパス
    """
```

### 9.2 コンポーネントビルダー

```python
def _create_vit_backbone(compile_mode: Optional[str] = None) -> ViT:
    """ViTバックボーンを作成"""

def _create_vit_neck(
    position_encoding: nn.Module,
    vit_backbone: ViT,
    enable_inst_interactivity: bool = False
) -> Sam3DualViTDetNeck:
    """特徴ピラミッドネックを作成"""

def _create_vl_backbone(
    vit_neck: Sam3DualViTDetNeck,
    text_encoder: VETextEncoder
) -> SAM3VLBackbone:
    """Vision-Languageバックボーンを作成"""

def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Transformerエンコーダーを作成"""

def _create_transformer_decoder() -> TransformerDecoder:
    """Transformerデコーダーを作成"""

def _create_segmentation_head(
    compile_mode: Optional[str] = None
) -> UniversalSegmentationHead:
    """セグメンテーションヘッドを作成"""

def _create_geometry_encoder() -> SequenceGeometryEncoder:
    """ジオメトリエンコーダーを作成"""

def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """テキストエンコーダーを作成"""
```

---

## 10. エージェントインターフェース

### 10.1 agent_inference()

`sam3/agent/agent_core.py`

MLLM改善を使用したマルチラウンドセグメンテーションのメインエントリポイント。

#### パラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `image_path` | str | 入力画像の場所 |
| `text_prompt` | str | 初期テキストクエリ |
| `send_generate_request` | Callable | LLMリクエスト関数 |
| `call_sam_service` | Callable | SAM3サービス関数 |

#### ツールハンドラー

- `segment_phrase()`: テキストでSAM3を呼び出し
- `examine_each_mask()`: 個別マスク検証
- `select_masks_and_return()`: 最終選択
- `report_no_mask()`: 失敗報告

#### サポート関数

```python
def save_debug_messages(messages: List[Dict], output_path: str) -> None:
    """デバッグメッセージを保存"""

def count_images(messages: List[Dict]) -> int:
    """メッセージ内の画像数をカウント"""

def cleanup_debug_files(output_path: str) -> None:
    """デバッグファイルをクリーンアップ"""

def visualize(image: Image, masks: List, boxes: List) -> Image:
    """結果を可視化"""
```

### 10.2 sam3_inference()

`sam3/agent/client_sam3.py`

テキストプロンプトを使用した画像セグメンテーション。

#### パラメータ

| パラメータ | 型 | 説明 |
|----------|------|------|
| `processor` | Sam3Processor | SAM3Processorインスタンス |
| `image_path` | str | 入力画像 |
| `text_prompt` | str | クエリテキスト |

#### 戻り値

```python
{
    "boxes": List[List[float]],  # 正規化された [0,1]
    "masks": List[str],           # RLEエンコード済み
    "scores": List[float],        # 信頼度スコア
    "image_size": Tuple[int, int] # 元の画像サイズ
}
```

### 10.3 call_sam_service()

`sam3/agent/client_sam3.py`

出力管理を含むエンドツーエンドワークフロー。

#### パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|----------|------|---------|------|
| `sam3_processor` | Sam3Processor | 必須 | モデルプロセッサ |
| `image_path` | str | 必須 | 入力画像 |
| `text_prompt` | str | 必須 | クエリ |
| `output_folder_path` | str | "sam3_output" | 結果ディレクトリ |

#### 処理ステップ

1. `remove_overlapping_masks()`によるマスクオーバーラップ除去
2. スコアベースのソート
3. RLE検証（最小5文字）
4. JSONシリアライゼーション
5. PNG可視化

#### 戻り値

保存されたJSON出力ファイルへのパス

### 10.4 LLMクライアント関数

`sam3/agent/client_llm.py`

```python
def send_generate_request(
    messages: List[Dict],
    server_url: str,
    model: str = "Llama-4-Maverick",
    max_tokens: int = 4096,
    api_key: str
) -> Dict:
    """
    OpenAI互換APIエンドポイント

    Args:
        messages: メッセージリスト
        server_url: APIサーバーURL
        model: モデル名
        max_tokens: 最大トークン数
        api_key: 認証キー

    Returns:
        生成レスポンス
    """

def send_direct_request(
    llm: vLLM,
    messages: List[Dict],
    sampling_params: vLLM_params
) -> Dict:
    """
    直接vLLM推論

    Args:
        llm: vLLMエンジン
        messages: メッセージリスト
        sampling_params: サンプリングパラメータ

    Returns:
        生成レスポンス
    """

def get_image_base64_and_mime(image_path: str) -> Tuple[str, str]:
    """
    画像をMIMEタイプ付きbase64に変換

    Returns:
        (base64_string, mime_type)
    """
```

---

## 11. ユーティリティ関数

### 11.1 ボックス操作

`sam3/model/box_ops.py`

#### フォーマット変換

```python
def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
    """中心形式から角形式に変換 [cx, cy, w, h] → [x0, y0, x1, y1]"""

def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    """角形式から中心形式に変換 [x0, y0, x1, y1] → [cx, cy, w, h]"""

def box_xywh_to_xyxy(x: Tensor) -> Tensor:
    """[x, y, w, h] → [x0, y0, x1, y1]"""

def box_xyxy_to_xywh(x: Tensor) -> Tensor:
    """[x0, y0, x1, y1] → [x, y, w, h]"""
```

#### 面積と検出

```python
def box_area(boxes: Tensor) -> Tensor:
    """
    ボックス面積の計算

    Args:
        boxes: [N, 4] テンソル [x0, y0, x1, y1]フォーマット

    Returns:
        面積 [N]
    """

def masks_to_boxes(masks: Tensor) -> Tensor:
    """
    バイナリマスクから境界ボックスを計算

    Args:
        masks: [N, H, W] ブールテンソル

    Returns:
        boxes: [N, 4] [x0, y0, x1, y1]
    """
```

#### IoU計算

```python
def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    ペアワイズIoU計算

    Args:
        boxes1: [N, 4]
        boxes2: [M, 4]

    Returns:
        iou: [N, M]
    """

def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    GIoU計算

    Returns:
        giou: [N, M]
    """
```

### 11.2 I/Oユーティリティ

`sam3/model/io_utils.py`

```python
def load_resource_as_video_frames(
    resource: Union[str, List, PIL.Image],
    resolution: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float]
) -> Tuple:
    """
    インテリジェントなリソースローダー

    対応するソースタイプ:
    - 動画ファイル: .mp4, .mov, .avi, .mkv, .webm
    - 画像シーケンス: 番号付きファイル (例: "001.jpg")
    - PIL画像リスト
    - ダミー動画: "<load-dummy-video-N>"

    Returns:
        (frames_tensor, video_height, video_width)
    """

def load_video_frames(
    video_path: str,
    resolution: int
) -> Iterator:
    """
    動画フレームを読み込み

    Yields:
        フレームテンソル
    """
```

#### キークラス

- **AsyncImageFrameLoader**: 非同期画像シーケンス読み込み
- **AsyncVideoFileLoaderWithTorchCodec**: GPU加速動画デコード
- **TorchCodecDecoder**: TorchCodecラッパー

### 11.3 マスク操作

`sam3/agent/helpers/masks.py`

```python
class BitMasks:
    """ビットマスク表現 [N, H, W]"""

    def __init__(self, tensor: Tensor):
        """
        Args:
            tensor: [N, H, W] ブールテンソル
        """

    def crop_and_resize(self, box: Tensor, size: int) -> 'BitMasks':
        """ボックスにクロップしてリサイズ"""

    def get_bounding_boxes(self) -> Tensor:
        """
        マスクから境界ボックスを抽出

        Returns:
            boxes: [N, 4] [x0, y0, x1, y1]
        """

class PolygonMasks:
    """ポリゴンマスク表現"""

    def __init__(self, polygons: List[List[float]]):
        """
        Args:
            polygons: [[x1, y1, x2, y2, ...], ...]
        """

    def area(self) -> Tensor:
        """ポリゴン面積を計算"""

class ROIMasks:
    """ROIマスク表現 [N, M, M]"""

    def to_bitmasks(self, size: Tuple[int, int]) -> BitMasks:
        """ビットマスクに変換"""
```

### 11.4 RLEエンコーディング

`sam3/agent/helpers/rle.py`

```python
def rle_encode(masks: Tensor) -> List[Dict]:
    """
    ブールマスクをRLE形式にエンコード（部分的にGPU上）

    Args:
        masks: [N, H, W] ブールテンソル

    Returns:
        RLE辞書のリスト {'size': [H, W], 'counts': str}
    """

def robust_rle_encode(masks: Tensor) -> List[Dict]:
    """GPU優先、CPUフォールバック付き"""

def ann_to_rle(
    annotation: Union[List, Dict],
    height: int,
    width: int
) -> Dict:
    """
    様々なアノテーション形式をRLEに変換
    """
```

### 11.5 マスクオーバーラップ除去

`sam3/agent/helpers/mask_overlap_removal.py`

```python
def mask_intersection(
    mask1: Union[np.ndarray, Dict],
    mask2: Union[np.ndarray, Dict]
) -> float:
    """
    ピクセル単位のオーバーラップを計算

    Returns:
        交差ピクセル数
    """

def mask_iom(mask1, mask2) -> float:
    """
    IoM = intersection / min(area1, area2)

    Returns:
        IoMスコア
    """

def remove_overlapping_masks(
    sample: Dict,
    iom_threshold: float = 0.3
) -> Dict:
    """
    信頼度スコアによる貪欲フィルタリング

    Args:
        sample: {masks, boxes, scores}を含む辞書
        iom_threshold: オーバーラップ閾値

    Returns:
        kept_indices, removed_indicesを追加した辞書
    """
```

### 11.6 距離変換

`sam3/model/edt.py`

```python
def edt_triton(x: Tensor) -> Tensor:
    """
    Tritonカーネルを使用したユークリッド距離変換

    Args:
        x: [B, H, W] バイナリテンソル

    Returns:
        距離マップ（同じ形状）

    Note:
        H100ハードウェアでOpenCVより約5.5倍高速
    """
```

### 11.7 可視化ユーティリティ

`sam3/visualization_utils.py` および `sam3/agent/helpers/visualizer.py`

#### 画像表示

```python
def show_img_tensor(img_tensor: Tensor) -> PIL.Image:
    """テンソルをPIL画像に変換して表示"""

def load_frame(frame_input: Union[np.ndarray, PIL.Image, str]) -> np.ndarray:
    """様々な入力形式からフレームを読み込み"""
```

#### バウンディングボックス可視化

```python
def plot_bbox(
    ax,
    boxes: Tensor,
    format: str = "XYXY"
) -> None:
    """
    matplotlibの軸にボックスをプロット

    Args:
        ax: matplotlib軸
        boxes: [N, 4] テンソル
        format: "XYXY" または "XYWH"
    """

def draw_box_on_image(
    image: PIL.Image,
    boxes: Tensor
) -> PIL.Image:
    """画像にボックスを描画"""
```

#### マスク可視化

```python
def plot_mask(
    ax,
    masks: Tensor,
    alpha: float = 0.5,
    colors=None
) -> None:
    """
    マスクを透明度付きでプロット

    Args:
        ax: matplotlib軸
        masks: [N, H, W] テンソル
        alpha: 透明度
        colors: 色のリスト
    """

def show_mask(
    ax,
    masks: Tensor,
    random_colors: bool = True
) -> None:
    """ランダムカラーでマスクを表示"""
```

#### プロンプト可視化

```python
def show_points(
    ax,
    points: Tensor,
    labels: Tensor
) -> None:
    """
    ポイントプロンプトを表示

    正（label=1）は緑、負（label=0）は赤
    """
```

#### レンダリングと保存

```python
def save_masklet_video(
    frames: List[np.ndarray],
    masks_per_frame: List,
    output_path: str
) -> None:
    """
    マスク付き動画を保存（ffmpeg再エンコード使用）

    Args:
        frames: フレームリスト
        masks_per_frame: フレームごとのマスクリスト
        output_path: 出力パス
    """

def generate_colors(
    num_colors: int,
    seed: Optional[int] = None
) -> List[Tuple]:
    """
    LAB空間でK-meansを使用した知覚的に均一な色を生成

    Args:
        num_colors: 生成する色の数
        seed: 再現性のためのシード

    Returns:
        RGB色タプルのリスト
    """
```

### 11.8 ロギング

`sam3/logger.py`

```python
class ColoredFormatter(logging.Formatter):
    """カラーフォーマット付きロギング"""

    def format(self, record: logging.LogRecord) -> str:
        """カラーフォーマットでログレコードをフォーマット"""

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    ロガーを作成

    Features:
    - カラーフォーマットされたコンソール出力
    - LOG_LEVEL環境変数サポート
    - 重複防止のため伝播を無効化

    Args:
        name: ロガー名
        level: ログレベル

    Returns:
        設定済みロガー
    """
```

---

## 12. 評価モジュール

### 12.1 CocoEvaluator

`sam3/eval/coco_eval.py`

#### 主要メソッド

```python
class CocoEvaluator:
    def __init__(
        self,
        coco_gt,
        iou_types: List[str],
        distributed: bool = False,
        is_coco: bool = True
    ):
        """
        COCO評価器を初期化

        Args:
            coco_gt: Ground truth COCOオブジェクト
            iou_types: ['bbox', 'segm', 'keypoints']
            distributed: 分散評価
            is_coco: COCOフォーマット
        """

    def update(self, predictions: Dict) -> None:
        """モデル予測を処理"""

    def synchronize_between_processes(self) -> None:
        """分散プロセス間で結果を収集"""

    def accumulate(self, img_ids: Optional[List] = None) -> None:
        """評価結果を集計"""

    def summarize(
        self,
        print_results: bool = True,
        rarity_bucket_avg: bool = False
    ) -> None:
        """サマリーメトリクスを計算して表示"""

    def compute_synced(self) -> Dict:
        """完全な評価パイプライン"""
```

#### サポートされるIoUタイプ

- `bbox`: バウンディングボックス評価
- `segm`: セグメンテーションマスク評価
- `keypoints`: キーポイント評価

---

## 13. 使用例

### 13.1 モデルの構築

```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 画像モデルを構築
model = build_sam3_image_model(
    bpe_path="path/to/bpe.model",
    device="cuda",
    checkpoint_path="path/to/checkpoint.pt",
    compile=True,
    eval_mode=True
)

# プロセッサを作成
processor = Sam3Processor(
    model=model,
    resolution=1008,
    device="cuda",
    confidence_threshold=0.5
)
```

### 13.2 画像セグメンテーション

```python
from PIL import Image

# 画像を読み込んで処理
image = Image.open("image.jpg")
processor.set_image(image)

# テキストプロンプトを追加
processor.set_text_prompt("cat")

# 結果を取得
results = processor._forward_grounding()
masks = results['masks']
boxes = results['boxes']
scores = results['scores']

# 結果を可視化
from sam3.visualization_utils import plot_mask, plot_bbox
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(image)
plot_mask(ax, masks, alpha=0.5)
plot_bbox(ax, boxes)
plt.show()
```

### 13.3 バッチ処理

```python
# 複数画像を処理
images = [Image.open(f"image_{i}.jpg") for i in range(5)]
processor.set_image_batch(images)
processor.set_text_prompt("person")

# バッチ結果を取得
results = processor.predict_inst_batch(images)

for i, result in enumerate(results):
    print(f"Image {i}: {len(result['masks'])} objects detected")
```

### 13.4 ジオメトリプロンプト付きセグメンテーション

```python
import torch

# 画像を設定
processor.set_image(image)

# バウンディングボックスプロンプトを追加
bbox = torch.tensor([[100, 100, 300, 300]])  # [x0, y0, x1, y1]
processor.add_geometric_prompt(bbox=bbox, label=1)

# ポイントプロンプトを追加
points = torch.tensor([[200, 200], [250, 250]])
processor.add_geometric_prompt(points=points, labels=torch.tensor([1, 1]))

# テキストプロンプトと組み合わせ
processor.set_text_prompt("cat")

# 結果を取得
results = processor._forward_grounding()
```

### 13.5 動画トラッキング

```python
from sam3.model_builder import build_tracker

# トラッカーを構築
tracker = build_tracker(
    apply_temporal_disambiguation=True,
    with_backbone=True
)

# 動画で初期化
video_info = {
    'height': 720,
    'width': 1280,
    'fps': 30
}
tracker.init_state("video.mp4", video_info)

# 最初のフレームでユーザーインタラクションを追加
tracker.add_new_points_or_box(
    frame_idx=0,
    points=torch.tensor([[640, 360]]),  # 中心点
    box=None
)

# 動画全体で伝播
tracker.propagate_in_video(forward=True, backward=True)

# 出力を取得
outputs = tracker._get_orig_video_res_output()

# 各フレームを処理
for frame_idx, output in enumerate(outputs):
    masks = output['masks']
    boxes = output['boxes']
    track_ids = output['track_ids']
    print(f"Frame {frame_idx}: {len(masks)} objects tracked")
```

### 13.6 インタラクティブ動画セグメンテーション

```python
# 動画プレディクタを構築
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor()

# セッションを開始
session_id = predictor.start_session("video.mp4")

# フレーム0にテキストプロンプトを追加
predictor.add_prompt(
    session_id=session_id,
    frame_idx=0,
    prompt_type="text",
    prompt_data="person wearing red shirt"
)

# 前方向に伝播
for frame_output in predictor.propagate_in_video(
    session_id=session_id,
    direction="forward"
):
    frame_idx = frame_output['frame_idx']
    masks = frame_output['masks']
    print(f"Processed frame {frame_idx}")

# フレーム10に改善ポイントを追加
predictor.add_prompt(
    session_id=session_id,
    frame_idx=10,
    prompt_type="point",
    prompt_data={'points': [[500, 400]], 'labels': [1]}
)

# 再伝播
for frame_output in predictor.propagate_in_video(
    session_id=session_id,
    direction="both"
):
    # 更新された結果を処理
    pass

# セッションを閉じる
predictor.close_session(session_id)
```

### 13.7 エージェントベースセグメンテーション

```python
from sam3.agent.agent_core import agent_inference
from sam3.agent.client_llm import send_generate_request
from sam3.agent.client_sam3 import call_sam_service

# LLMループでエージェントを実行
results = agent_inference(
    image_path="complex_scene.jpg",
    text_prompt="find all cats in the image",
    send_generate_request=lambda msgs: send_generate_request(
        messages=msgs,
        server_url="http://localhost:8000/v1",
        model="Llama-4-Maverick",
        api_key="your-api-key"
    ),
    call_sam_service=lambda img_path, text: call_sam_service(
        sam3_processor=processor,
        image_path=img_path,
        text_prompt=text,
        output_folder_path="agent_output"
    )
)

# エージェントは以下を実行します:
# 1. LLMにプロンプトを送信
# 2. セグメンテーションのためSAM3を呼び出し
# 3. LLMで結果を検証
# 4. 反復的に改善
```

### 13.8 カスタム評価

```python
from sam3.eval.coco_eval import CocoEvaluator
from pycocotools.coco import COCO

# Ground truthを読み込み
coco_gt = COCO("annotations/instances_val2017.json")

# 評価器を作成
evaluator = CocoEvaluator(
    coco_gt=coco_gt,
    iou_types=['bbox', 'segm'],
    distributed=False
)

# モデル予測を実行
for image_id in coco_gt.getImgIds():
    # 画像を読み込んで推論
    image_info = coco_gt.loadImgs(image_id)[0]
    image = Image.open(image_info['file_name'])

    processor.set_image(image)
    processor.set_text_prompt("object")
    results = processor._forward_grounding()

    # COCOフォーマットに変換
    predictions = {
        'image_id': image_id,
        'boxes': results['boxes'].cpu().numpy(),
        'masks': results['masks'].cpu().numpy(),
        'scores': results['scores'].cpu().numpy()
    }

    # 評価器を更新
    evaluator.update({image_id: predictions})

# 評価を実行
evaluator.accumulate()
evaluator.summarize()
```

### 13.9 複数マスク出力

```python
# 複数マスク出力を有効にしたモデルを構築
model = build_sam3_image_model(
    device="cuda",
    checkpoint_path="checkpoint.pt",
    eval_mode=True
)
model.multimask_output = True

processor = Sam3Processor(model=model)

# 単一ポイントプロンプトで複数マスクを取得
processor.set_image(image)
processor.add_geometric_prompt(
    points=torch.tensor([[250, 250]]),
    labels=torch.tensor([1])
)

results = processor._forward_grounding()

# 各オブジェクトに対して複数のマスク候補が返されます
masks = results['masks']  # [N, 3, H, W] - 3つのマスク候補
scores = results['scores']  # [N, 3] - 各候補のスコア

# 最高スコアのマスクを選択
best_masks = masks[torch.arange(len(masks)), scores.argmax(dim=1)]
```

### 13.10 モデルコンパイルと最適化

```python
# Torch Compileを有効にしてモデルを構築
model = build_sam3_image_model(
    device="cuda",
    checkpoint_path="checkpoint.pt",
    compile=True,  # Torch Compileを有効化
    eval_mode=True
)

processor = Sam3Processor(model=model)

# ウォームアップ実行（コンパイル用）
dummy_image = torch.randn(3, 1008, 1008).cuda()
processor.set_image(dummy_image)
processor.set_text_prompt("warmup")
_ = processor._forward_grounding()

# 実際の推論（最適化済み）
processor.set_image(real_image)
processor.set_text_prompt("cat")
results = processor._forward_grounding()
```

---

## まとめ

このSAM3 APIリファレンスは以下をカバーしています:

- **48以上のメインクラス**とメソッドシグネチャ
- **80以上のユーティリティ関数**（ヘルパーモジュール全体）
- すべての主要コンポーネントの**設定パラメータ**
- 一般的なワークフローの**データフロー例**
- すべての主要関数の**出力フォーマット**
- **エラーハンドリング**パターンと制約

SAM3 APIは、明確に定義されたレイヤーに整理されています:

1. **高レベルインターフェース** (Processor, Predictorクラス)
2. **モデルアーキテクチャ** (Image/Videoモデル, Backbones, Transformers)
3. **コンポーネントビルダー** (Encoders, Decoders, Necks)
4. **ユーティリティ関数** (Box/Mask操作, I/O, 可視化)
5. **評価ツール** (COCO, Video-VIS評価)

すべてのコンポーネントはPyTorchベースで、GPU加速、分散コンピューティング、パフォーマンス最適化のためのTorchコンパイルをサポートしています。

---

## 参考リンク

- **公式リポジトリ**: https://github.com/facebookresearch/sam3
- **論文**: Segment Anything Model 3 (SAM 3)
- **Hugging Face**: モデルチェックポイント
- **PyTorch**: https://pytorch.org/

---

## ライセンス

SAM3は、Meta Researchによって開発され、Apacheライセンスの下で提供されています。詳細はリポジトリのLICENSEファイルを参照してください。
