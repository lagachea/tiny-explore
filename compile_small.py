from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict
from export_model import export_model

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
