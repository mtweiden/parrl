from gymnasium import Env
from gymnasium import make

from torch import tensor

from parrl.networks.noisy_layer import NoisyLinear



class TestNoisyLinear:

    def test_constructor(self) -> None:
        layer = NoisyLinear(2, 3)
        assert layer.weight.shape == (3, 2)
        assert layer.bias.shape == (3,)

    def test_forward(self) -> None:
        layer = NoisyLinear(2, 3)
        x = tensor([[1.0, 2.0]])
        y = layer(x)
        assert y.shape == (1, 3)
