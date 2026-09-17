import json
from pathlib import Path


def get_vlm_model(config):
    vlm_name = config.framework.qwenvl.base_vlm
    local_config = Path(vlm_name) / "config.json"
    if local_config.is_file():
        # A pinned snapshot may be stored under a neutral directory name.
        model_type = json.loads(local_config.read_text())["model_type"]
    elif "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():
        model_type = "qwen2_5_vl"
    elif "Qwen3-VL" in vlm_name:
        model_type = "qwen3_vl"
    elif "florence" in vlm_name.lower():
        model_type = "florence2"
    else:
        model_type = None

    if model_type == "qwen2_5_vl":
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    if model_type == "qwen3_vl":
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    if model_type == "florence2":
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    raise NotImplementedError(f"VLM model {vlm_name} (model_type={model_type!r}) not implemented")
