###########################################################################################
# Radial basis and cutoff
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import numpy as np
import torch
from e3nn.util.jit import compile_mode


def _softplus_inverse(x: float) -> float:
    value = torch.tensor(x, dtype=torch.get_default_dtype())
    return torch.log(torch.expm1(value)).item()


PAIR_A_INIT = 1.5
PAIR_B_INIT = -2.25


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


@compile_mode("script")
class ChebyshevBasis(torch.nn.Module):
    """
    Chebyshev_Basis T_0=1 T_1=x T_n+1=2x*T_n-T_n-1
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()
        self.num_basis = num_basis

        bessel_weights = (
            torch.ones(num_basis, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x_s = x / self.r_max - 2
        T_n = torch.zeros(x_s.shape[0], self.num_basis, dtype=torch.get_default_dtype())
        if self.num_basis >= 1:
            T_n[:, 0] = 1.0
        if self.num_basis >= 2:
            T_n[:, 1] = x_s
        for n in range(2, self.num_basis):
            T_n[:, n] = 2.0 * x_s * T_n[:, n - 1] - T_n[:, n - 2]

        return T_n

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )



@compile_mode("script")
class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis=128, trainable=False):
        super().__init__()
        gaussian_weights = torch.linspace(
            start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype()
        )
        if trainable:
            self.gaussian_weights = torch.nn.Parameter(
                gaussian_weights, requires_grad=True
            )
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


@compile_mode("script")
class PairwiseTanhChebyshevBasis(torch.nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_basis: int,
        num_channels: int,
        num_l_channels: int = 1,
        epsilon: float = 1e-6,
        pair_scaling: str = "element_channel",
    ):
        super().__init__()
        self.num_basis = num_basis
        self.num_elements = num_elements
        self.num_channels = num_channels
        self.num_l_channels = num_l_channels
        self.epsilon = epsilon
        self.pair_scaling = pair_scaling
        self.use_scaling = pair_scaling != "none"
        self.element_channel_scaling = pair_scaling in ("element_channel", "element_channel_l")
        self.scaling_with_l = pair_scaling in ("element_l", "element_channel_l")

        if pair_scaling not in ("none", "element", "element_channel", "element_l", "element_channel_l"):
            raise ValueError(
                "pair_scaling must be one of: none, element, element_channel, element_l, element_channel_l"
            )

        if self.use_scaling:
            init_a_raw = _softplus_inverse(PAIR_A_INIT - epsilon)
            if self.element_channel_scaling and self.scaling_with_l:
                a_shape = (num_elements, num_elements, num_channels, num_l_channels)
            elif self.element_channel_scaling:
                a_shape = (num_elements, num_elements, num_channels)
            elif self.scaling_with_l:
                a_shape = (num_elements, num_elements, num_l_channels)
            else:
                a_shape = (num_elements, num_elements)

            self.a_raw = torch.nn.Parameter(
                torch.full(
                    a_shape,
                    init_a_raw,
                    dtype=torch.get_default_dtype(),
                )
            )
            self.b_raw = torch.nn.Parameter(
                torch.full(
                    a_shape,
                    PAIR_B_INIT,
                    dtype=torch.get_default_dtype(),
                )
            )
        else:
            self.register_parameter("a_raw", None)
            self.register_parameter("b_raw", None)

        self.register_buffer(
            "orders",
            torch.arange(num_basis, dtype=torch.get_default_dtype()),
        )

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        sender_attrs: torch.Tensor,  # [n_edges, n_elements]
        receiver_attrs: torch.Tensor,  # [n_edges, n_elements]
    ) -> torch.Tensor:
        edge_lengths = edge_lengths.to(dtype=torch.get_default_dtype())
        if edge_lengths.shape[0] == 0:
            if self.scaling_with_l:
                return edge_lengths.new_zeros(
                    (0, self.num_channels * self.num_basis * self.num_l_channels)
                )
            return edge_lengths.new_zeros((0, self.num_channels * self.num_basis))

        if self.use_scaling:
            a_raw = 0.5 * (self.a_raw + self.a_raw.transpose(0, 1))
            b = 0.5 * (self.b_raw + self.b_raw.transpose(0, 1))
            a = torch.nn.functional.softplus(a_raw) + self.epsilon

            sender_attrs = sender_attrs.to(dtype=a.dtype)
            receiver_attrs = receiver_attrs.to(dtype=a.dtype)

            if self.element_channel_scaling and self.scaling_with_l:
                pair_a = torch.einsum("ea,eb,abcl->ecl", sender_attrs, receiver_attrs, a)
                pair_b = torch.einsum("ea,eb,abcl->ecl", sender_attrs, receiver_attrs, b)
            elif self.element_channel_scaling:
                pair_a = torch.einsum("ea,eb,abc->ec", sender_attrs, receiver_attrs, a)
                pair_b = torch.einsum("ea,eb,abc->ec", sender_attrs, receiver_attrs, b)
            elif self.scaling_with_l:
                pair_a_scalar = torch.einsum("ea,eb,abl->el", sender_attrs, receiver_attrs, a)
                pair_b_scalar = torch.einsum("ea,eb,abl->el", sender_attrs, receiver_attrs, b)
                pair_a = pair_a_scalar.unsqueeze(1).expand(-1, self.num_channels, -1)
                pair_b = pair_b_scalar.unsqueeze(1).expand(-1, self.num_channels, -1)
            else:
                pair_a_scalar = torch.einsum("ea,eb,ab->e", sender_attrs, receiver_attrs, a)
                pair_b_scalar = torch.einsum("ea,eb,ab->e", sender_attrs, receiver_attrs, b)
                pair_a = pair_a_scalar.unsqueeze(-1).expand(-1, self.num_channels)
                pair_b = pair_b_scalar.unsqueeze(-1).expand(-1, self.num_channels)
        else:
            if self.scaling_with_l:
                pair_a = edge_lengths.new_ones((edge_lengths.shape[0], self.num_channels, self.num_l_channels))
                pair_b = edge_lengths.new_zeros((edge_lengths.shape[0], self.num_channels, self.num_l_channels))
            else:
                pair_a = edge_lengths.new_ones((edge_lengths.shape[0], self.num_channels))
                pair_b = edge_lengths.new_zeros((edge_lengths.shape[0], self.num_channels))

        rho = edge_lengths.view(-1, 1, 1) if self.scaling_with_l else edge_lengths.view(-1, 1)
        transformed = pair_a * rho + pair_b
        bounded = torch.tanh(transformed).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if self.scaling_with_l:
            cheby = torch.cos(
                torch.arccos(bounded).unsqueeze(-1) * self.orders.view(1, 1, 1, -1)
            )
            cheby = cheby.permute(0, 1, 3, 2)  # [n_edges, C, K, L]
            return cheby.reshape(
                cheby.shape[0], self.num_channels * self.num_basis * self.num_l_channels
            )

        cheby = torch.cos(
            torch.arccos(bounded).unsqueeze(-1) * self.orders.view(1, 1, -1)
        )
        return cheby.reshape(cheby.shape[0], self.num_channels * self.num_basis)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_elements={self.num_elements}, "
            f"num_channels={self.num_channels}, num_basis={self.num_basis}, "
            f"num_l_channels={self.num_l_channels}, "
            f"epsilon={self.epsilon}, pair_scaling={self.pair_scaling})"
        )


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"
