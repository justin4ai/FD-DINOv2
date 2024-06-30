import inspect
from src.pytorch_fd.dino import DINOv2Encoder, Encoder

def load_encoder(model_name, device, **kwargs):
    """Load feature extractor"""

    model_cls = DINOv2Encoder

    signature = inspect.signature(model_cls.setup)
    arguments = list(signature.parameters.keys())


    arguments = arguments[1:] # Omit `self` arg

    encoder = model_cls(**{arg: kwargs[arg] for arg in arguments if arg in kwargs})
    encoder.name = model_name

    assert isinstance(encoder, Encoder), "Can only get representations with Encoder subclasses!"

    return encoder.to(device)