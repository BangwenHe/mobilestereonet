import torch

from models import MSNet2D


if __name__ == "__main__":
    MAX_DISP = 192
    model_path = "checkpoints/MSNet2D_SF_DS_KITTI2015.ckpt"

    model = MSNet2D(MAX_DISP)
    
    state_dict = torch.load(model_path)
    pretrained_dict = {key.replace("module.", ""): value for key, value in state_dict['model'].items()}
    model.eval()

    left_image = torch.randn((1, 3, 640, 480))
    right_image = torch.randn((1, 3, 640, 480))

    torch.onnx.export(
        model,
        (left_image, right_image),
        "predictions/msnet.onnx",
        opset_version=11
    )

    # scripted_model = torch.jit.script(model)
    # scripted_model._save_for_lite_interpreter("predictions/msnet.pt")
