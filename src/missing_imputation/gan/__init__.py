from .clr import clr_transform, inverse_clr_transform

__all__ = [
    "clr_transform",
    "inverse_clr_transform",
    "DeepMicroGenGenerator",
    "DeepMicroGenDiscriminator",
    "DeepMicroGenGAN",
    "GANDataset",
    "build_gan_panel",
]


def __getattr__(name: str):
    if name in ("DeepMicroGenGenerator", "DeepMicroGenDiscriminator", "DeepMicroGenGAN"):
        from . import model as _model

        return getattr(_model, name)
    if name in ("GANDataset", "build_gan_panel"):
        from . import data as _data

        return getattr(_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
