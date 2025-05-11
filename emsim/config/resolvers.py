from omegaconf import OmegaConf


def array_length(array_path: str) -> int:
    """Returns the length of an array at the given config path."""
    cfg = OmegaConf.get_current()
    array = OmegaConf.select(cfg, array_path)
    if array is None:
        raise ValueError(f"Path '{array_path}' not found in config")
    return len(array)


def backbone_encoder_output_stride() -> int:
    """Returns the default backbone encoder output stride based on number of layers"""
    path = "model.backbone.encoder.layers"
    n_encoder_layers = array_length(path)
    return 2 ** n_encoder_layers


def n_aux_losses() -> int:
    """Returns the default number of aux losses based on number of decoder layers"""
    path = "model.transformer.decoder.layers"
    n_decoder_layers = array_length(path)
    return n_decoder_layers - 1


def register_resolvers():
    OmegaConf.register_resolver("array_length", array_length)
    OmegaConf.register_resolver("backbone_encoder_output_stride", backbone_encoder_output_stride)
    OmegaConf.register_resolver("n_aux_losses", n_aux_losses)
