from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict
from export_model import export_model
import torch.nn as nn

class SmallModel(nn.Module):
    def __init__(self, classes):
        super(SmallModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(64 * 54 * 54, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# make a tinygrad implementation to compile
# class SmallTiny:

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    model = None
    inputs: Tensor = Tensor.randn(1, 3, 640, 640) # model input likeness

    # ensure safetensors and model works
    # state_dict = safe_load("small.safetensos")

    prg, _, _, state = export_model(model, Device.DEFAULT.lower(), inputs, model_name="small"
    )

    # save
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
        text_file.write(prg)
