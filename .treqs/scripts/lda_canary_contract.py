from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_BASE_COMMIT = "06e6a274a9086cc26635a9fe663866335eb30fc5"

BASE_MODEL_ID = "Wayer2/LDA-robocasa"
BASE_MODEL_REVISION = "811d14d8c22d3e98021c035948118143f53dd312"
BASE_CHECKPOINT_NAME = "LDA-robocasa.pt"
QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DINO_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINO_MODEL_REVISION = "114c1379950215c8b35dfcd4e90a5c251dde0d32"

DATASET_PATH = ROOT / "playground" / "demo_data" / "sim_pick_place"
EXPECTED_EPISODES = 4

PUBLICATION_REPO_ID = "reproducible-ai/harness-test-lda-robocasa-issue-5"
PUBLICATION_VERSION = "robocasa-demo-canary-0.0.1"

ARTIFACT_ROOT = ROOT / "artifacts" / "lda-robocasa-canary"
INPUT_ROOT = ARTIFACT_ROOT / "inputs"
BASE_SNAPSHOT = INPUT_ROOT / "LDA-robocasa"
BASE_CHECKPOINT = BASE_SNAPSHOT / "checkpoints" / BASE_CHECKPOINT_NAME
QWEN_SNAPSHOT = INPUT_ROOT / "Qwen3-VL-4B-Instruct"
PRETRAINED_ROOT = INPUT_ROOT / "pretrained"
DINO_SNAPSHOT = PRETRAINED_ROOT / "dinov3-vits16-pretrain-lvd1689m"
DINO_CONFIG = {
    "architectures": ["DINOv3ViTModel"],
    "attention_dropout": 0.0,
    "drop_path_rate": 0.0,
    "hidden_act": "gelu",
    "hidden_size": 384,
    "image_size": 224,
    "initializer_range": 0.02,
    "intermediate_size": 1536,
    "key_bias": False,
    "layer_norm_eps": 1e-5,
    "layerscale_value": 1.0,
    "mlp_bias": True,
    "model_type": "dinov3_vit",
    "num_attention_heads": 6,
    "num_channels": 3,
    "num_hidden_layers": 12,
    "num_register_tokens": 4,
    "patch_size": 16,
    "pos_embed_jitter": None,
    "pos_embed_rescale": 2.0,
    "pos_embed_shift": None,
    "proj_bias": True,
    "query_bias": True,
    "rope_theta": 100.0,
    "torch_dtype": "float32",
    "use_gated_mlp": False,
    "value_bias": True,
}
DINO_PREPROCESSOR_CONFIG = {
    "crop_size": None,
    "data_format": "channels_first",
    "default_to_square": True,
    "do_center_crop": None,
    "do_convert_rgb": None,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "DINOv3ViTImageProcessorFast",
    "image_std": [0.229, 0.224, 0.225],
    "resample": 2,
    "rescale_factor": 1 / 255,
    "size": {"height": 224, "width": 224},
}
INPUT_MANIFEST_PATH = ARTIFACT_ROOT / "input-manifest.json"

RUN_ROOT = ARTIFACT_ROOT / "training"
RUN_ID = "robocasa-demo-canary"
RUN_DIR = RUN_ROOT / RUN_ID
TRAINED_CHECKPOINT = RUN_DIR / "checkpoints" / "steps_1_pytorch_model.pt"
EVALUATION_PATH = RUN_DIR / "evaluation.json"
RELEASE_ROOT = ARTIFACT_ROOT / "release"
RELEASE_CHECKPOINT = RELEASE_ROOT / "checkpoints" / "LDA-robocasa-treqs-canary.pt"
