from pathlib import Path
from typing import Callable
from tinygrad import Tensor, nn
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict, safe_save, get_state_dict
from export_model import export_model
from json import load


class AlexNet:
    def __init__(self, classes):
        self.features: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4),
            Tensor.relu,
            lambda x: x.max_pool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2),
            Tensor.relu,
            lambda x: x.max_pool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
            Tensor.relu,
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),
            Tensor.relu,
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            Tensor.relu,
            lambda x: x.max_pool2d(kernel_size=3, stride=2),
        ]

        self.classifier: list[Callable[[Tensor], Tensor]] = [
            lambda x: x.flatten(1),
            nn.Linear(9216, 4096),
            Tensor.relu,
            nn.Linear(4096, 4096),
            Tensor.relu,
            nn.Linear(4096, classes),
        ]

    def forward(self, x: Tensor) -> Tensor:
        return x.sequential([*self.features, *self.classifier])


class SmallModel:
    def __init__(self, classes):
        self.features: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(3, 32, 3),
            Tensor.relu,
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3),
            Tensor.relu,
            Tensor.max_pool2d,
        ]

        self.classifier: list[Callable[[Tensor], Tensor]] = [
            lambda x: x.flatten(1),
            nn.Linear(64 * 54 * 54, classes),
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential([*self.features, *self.classifier])


# list models json/safetensors
# match name with class

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    classes = [
        "Apple_Black_rot",
        "Apple_healthy",
        "Apple_rust",
        "Apple_scab",
        "Grape_Black_rot",
        "Grape_Esca",
        "Grape_healthy",
        "Grape_spot",
    ]
    dirname = Path(__file__).parent / "models"

    models = {
        SmallModel(len(classes)): Tensor.randn(1, 3, 224, 224),
        AlexNet(len(classes)): Tensor.randn(1, 3, 227, 227),
    }
    safetensors = [
        safe_load(dirname / "SmallModel.safetensors"),
        safe_load(dirname / "AlexNet.safetensors"),
    ]

    for (model, input), state in zip(models.items(), safetensors):
        # model_state = get_state_dict(model)
        # print(f"{model_state=}")
        # for k,v in model_state.items():
        # print(f"{k=}, {v.shape=}")
        # print(k)

        # print(f"{state=}")
        # for k,v in state.items():
        # print(f"{k=}, {v.shape=}")
        # print(k)

        load_state_dict(model, state)
        # print(f"{loaded=}")
        model_name = model.__class__.__name__
        prg, inp_sizes, out_sizes, state = export_model(
            model, Device.DEFAULT.lower(), input, model_name=model_name
        )

        # safe_save(state, (dirname / f"{model_name}.safetensors").as_posix())

        with open(dirname / f"{model_name}.js", "w") as text_file:
            text_file.write(prg)
