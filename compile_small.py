from pathlib import Path
from typing import Callable
from tinygrad import Tensor, nn
from tinygrad.nn.state import safe_save
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict
from export_model import export_model
# import torch.nn as nn
from json import load

# class SmallModel(nn.Module):
#     def __init__(self, classes):
#         super(SmallModel, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=(3, 3)),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=(3, 3)),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(1),
#             nn.Dropout(0.5),
#             nn.Linear(64 * 54 * 54, classes)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# make a tinygrad implementation to compile
class SmallTiny:
    def __init__(self, classes):
        self.features: list[Tensor] = [
            nn.Conv2d(3,32, 3), Tensor.relu, Tensor.max_pool2d,
            nn.Conv2d(32,64, 3), Tensor.relu, Tensor.max_pool2d,
        ]

        self.classifier: list[Tensor] = [lambda x: x.flatten(1), nn.Linear(64 * 54 * 54, classes)]
    
    def __call__(self, x:Tensor) -> Tensor:
        x = x.sequential(self.features)
        x = x.sequential(self.classifier)
        return self.classifier(x)

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"

    classes = None
    with open("SmallModel-images_dataset-Epch_10-Acc_92.json", "r") as f:
        classes = load(f)

    model = SmallTiny(len(classes))
    print(model.features, model.classifier)

    state_dict = safe_load("SmallModel-images_dataset-Epch_10-Acc_92.safetensors")
    ret = load_state_dict(model, state_dict)

    inputs: Tensor = Tensor.randn(1, 3, 640, 640) # model input likeness

    print(model.features, model.classifier)
    exit(0)

    # ensure safetensors and model works
    # state_dict = safe_load("small.safetensos")

    prg, _, _, state = export_model(model, Device.DEFAULT.lower(), inputs, model_name="small"
    )

    # save
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
        text_file.write(prg)
