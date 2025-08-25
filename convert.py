from pathlib import Path
from tinygrad.nn.state import safe_load, safe_save

def get_models(dir: str = "models"):
    weights: list[Path] = list(Path(dir).rglob("*.safetensors"))
    config: list[Path] = list(Path(dir).rglob("*.json"))
    return {key: value for key,value in zip(config,weights)}

def convert_small():
    model_paths = [
        "models/SmallModel-Grape_dataset-Epch:10-Acc:97.safetensors",
        "models/SmallModel-images_dataset-Epch:10-Acc:92.safetensors",
    ]

    for path in model_paths:
        state = safe_load(path)

        match = {
            "classifier.1.weight": "classifier.2.weight",
            "classifier.1.bias": "classifier.2.bias",
        }

        for k, v in match.items():
            state[k] = state.pop(v)
        safe_save(state, path)

def convert_alex():
    model_paths = [
        "models/AlexNet-images_dataset-Epch:10-Acc:96.safetensors",
        "models/AlexNet-Apple_dataset-Epch:10-Acc:91.safetensors",
        "models/AlexNet-images_dataset-Epch:10-Acc:93.safetensors",
    ]

    for path in model_paths:
        state = safe_load(path)

        match = {
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            "features.6.weight": "conv3.weight",
            "features.6.bias": "conv3.bias",
            "features.8.weight": "conv4.weight",
            "features.8.bias": "conv4.bias",
            "features.10.weight": "conv5.weight",
            "features.10.bias": "conv5.bias",
            "classifier.1.weight": "fc1.weight",
            "classifier.1.bias": "fc1.bias",
            "classifier.3.weight": "fc2.weight",
            "classifier.3.bias": "fc2.bias",
            "classifier.5.weight": "fc3.weight",
            "classifier.5.bias": "fc3.bias",
        }

        for k, v in match.items():
            state[k] = state.pop(v)
        safe_save(state, path)

if __name__ == "__main__":
    convert_alex()
    convert_small()
    # models = get_models()
    # print(f"{models=}")
