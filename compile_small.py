from pathlib import Path
from typing import Callable
from tinygrad import Tensor, nn
from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict
from export_model import export_model
from json import load


class SmallTiny:
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
        return x.sequential(self.features).sequential(self.classifier)


if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    classes = None
    with open("SmallModel-images_dataset-Epch_10-Acc_92.json", "r") as f:
        classes = load(f)

    model = SmallTiny(len(classes))
    model_state = get_state_dict(model)
    # print(f"{model=}")

    state = safe_load("net.safetensors")
    # print(f"{state=}")

    loaded = load_state_dict(model, state)
    # print(f"{loaded=}")

    inputs: Tensor = Tensor.randn(1, 3, 224, 224)  # model input likeness
    prg, inp_sizes, out_sizes, state = export_model(
        model, Device.DEFAULT.lower(), inputs, model_name="smalltiny"
    )

    # save
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
        text_file.write(prg)
