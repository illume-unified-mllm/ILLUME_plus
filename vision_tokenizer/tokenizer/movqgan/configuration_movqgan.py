""" MoVQ model configuration """

from typing import List

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MoVQConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MoVQ`]. It is used to instantiate an video movq
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a configuration to the VQ model presented in  paper.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        codebook_size (`int`, *optional*, defaults to 32768):
            Codebook size of the VQ model.
        embed_dim (`int`, *optional*, defaults to 4):
            Dimension of the quantized vector in codebook.
        z_channels (`int`, *optional*, defaults to 4):
            Dimension of the output channel of encoder and the input channel of decoder
        double_z (`bool`, *optional*, defaults to False):
            Whether double the output dim of the encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Input channel of encoder.
        out_channels (`int`, *optional*, defaults to 3):
            Output channel of decoder.
        ch (`int`, *optional*, defaults to 256):
            Basic channel number of the intermediate blocks.
        ch_mult (`List[int]`, *optional*, defaults to `[1, 2, 2, 4]`):
            Channel scaling factor of the intermediate blocks.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Residual block number in each stage.
        attn_resolutions (`List[int]`, *optional*, defaults to 3):
            Stage indices to apply attention.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability.

    ```python
    >>> from transformers import MoVQ, MoVQConfig

    >>> # Initializing a video VQ model of  configuration
    >>> configuration = MoVQConfig()

    >>> # Initializing a model from the  VQ model style configuration
    >>> model = MoVQModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "MoVQ"

    def __init__(
        self,
        codebook_size: int = 32768,
        embed_dim: int = 4,
        z_channels: int = 4,
        double_z: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        ch: int = 256,
        ch_mult: List[int] = [1, 2, 2, 4],
        num_res_blocks: int = 2,
        attn_resolutions: List[int] = [3],
        dropout: float = 0.0,
        use_dc_up_down_blocks=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.z_channels = z_channels
        self.double_z = double_z
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ch = ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.use_dc_up_down_blocks = use_dc_up_down_blocks
