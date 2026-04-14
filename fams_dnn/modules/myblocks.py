###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from ptagnn.tools.scatter import scatter_sum

from .irreps_tools import (
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from .radial import (
    BesselBasis,
    GaussianBasis,
    PolynomialCutoff,
    ChebyshevBasis,
    PairwiseTanhChebyshevBasis,
)
from .symmetric_contraction import SymmetricContraction


@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps("0e"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate: Optional[Callable]
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=o3.Irreps("0e")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]




@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        num_elements: Optional[int] = None,
        num_pair_channels: Optional[int] = None,
        pair_scaling: str = "element_channel",
        radial_with_l: bool = False,
        radial_version: str = "v1",
        max_ell: Optional[int] = None,
    ):
        super().__init__()
        self.radial_type = radial_type
        self.use_pairwise_transform = radial_type == "pairwise_tanh_chebyshev"
        self.radial_with_l = radial_with_l
        self.radial_version = radial_version
        self.num_pair_channels = num_pair_channels
        if pair_scaling in ("element_l", "element_channel_l") and not radial_with_l:
            raise ValueError("pair_scaling modes with _l require radial_with_l to be enabled")
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebyshevBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "pairwise_tanh_chebyshev":
            if num_elements is None or num_pair_channels is None:
                raise ValueError(
                    "num_elements and num_pair_channels are required for pairwise_tanh_chebyshev"
                )
            self.bessel_fn = PairwiseTanhChebyshevBasis(
                num_elements=num_elements,
                num_basis=num_bessel,
                num_channels=num_pair_channels,
                num_l_channels=max_ell + 1 if radial_with_l else 1,
                pair_scaling=pair_scaling,
            )
        else:
            raise ValueError(f"Unknown radial_type: {radial_type}")
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        if self.radial_with_l:
            if num_pair_channels is None:
                raise ValueError("num_pair_channels is required when radial_with_l is enabled")
            base_out_dim = num_bessel * num_pair_channels
        else:
            base_out_dim = (
                num_bessel * num_pair_channels
                if self.use_pairwise_transform
                else num_bessel
            )
        if self.radial_with_l:
            if max_ell is None:
                raise ValueError("max_ell is required when radial_with_l is enabled")
            if max_ell < 0:
                raise ValueError("max_ell must be non-negative")
            self.num_l_channels = max_ell + 1
        else:
            self.num_l_channels = 1
        self.register_buffer(
            "l_powers",
            torch.arange(self.num_l_channels, dtype=torch.get_default_dtype()),
        )
        self.base_out_dim = base_out_dim
        self.out_dim = base_out_dim * self.num_l_channels

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        sender_attrs: Optional[torch.Tensor] = None,  # [n_edges, n_elements]
        receiver_attrs: Optional[torch.Tensor] = None,  # [n_edges, n_elements]
    ):
        if self.use_pairwise_transform:
            if sender_attrs is None or receiver_attrs is None:
                raise ValueError("sender_attrs and receiver_attrs are required for pairwise_tanh_chebyshev")
            radial = self.bessel_fn(edge_lengths, sender_attrs, receiver_attrs)
        else:
            radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        radial = radial * cutoff
        if not self.radial_with_l:
            return radial

        if self.use_pairwise_transform:
            if getattr(self.bessel_fn, "scaling_with_l", False):
                radial = radial.reshape(
                    radial.shape[0],
                    self.bessel_fn.num_channels,
                    self.bessel_fn.num_basis,
                    self.num_l_channels,
                )
            else:
                radial = radial.reshape(
                    radial.shape[0], self.bessel_fn.num_channels, self.bessel_fn.num_basis
                ).unsqueeze(-1).expand(-1, -1, -1, self.num_l_channels)
        else:
            radial = radial.unsqueeze(1).unsqueeze(-1).expand(
                -1, self.num_pair_channels, -1, self.num_l_channels
            )

        if self.radial_version == "v2":
            # Group features by l so each l block can be routed only to its matching Y_lm block.
            radial = radial.permute(0, 3, 1, 2)  # [n_edges, L, C, K]
            return radial.reshape(radial.shape[0], self.out_dim)

        scaled_lengths = (
            edge_lengths / self.cutoff_fn.r_max
        ).to(dtype=radial.dtype).clamp(min=1e-12)
        l_scales = torch.pow(scaled_lengths, self.l_powers.view(1, 1, 1, -1))
        radial = radial * l_scales
        return radial.reshape(radial.shape[0], self.out_dim)


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        else:
            return self.linear(node_feats)

@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        radial_MLP: Optional[List[int]] = None,
        radial_version: str = "v1",
        use_self_connection: bool = True,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.radial_version = radial_version
        self.use_self_connection = use_self_connection
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


@compile_mode("script")
class GroupedFullyConnectedNet(torch.nn.Module):
    def __init__(self, num_groups: int, hs: List[int]):
        super().__init__()
        self.num_groups = num_groups
        self.hs = hs
        self.weights = torch.nn.ParameterList()
        for h_in, h_out in zip(hs, hs[1:]):
            weight = torch.randn(
                num_groups,
                h_in,
                h_out,
                dtype=torch.get_default_dtype(),
            )
            self.weights.append(torch.nn.Parameter(weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for h_in, weight in zip(self.hs, self.weights):
            scale = float(h_in) ** 0.5
            x = torch.einsum("bgi,gih->bgh", x, weight / scale)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(num_groups={self.num_groups}, hs={self.hs})"



@compile_mode("script")
class Residual_InteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        ## MLP(hI)
        self.linear1 = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.irreps_out = self.target_irreps

        if self.radial_version == "v2":
            self.edge_attr_slices = []
            edge_offset = 0
            for mul, ir_edge in self.edge_attrs_irreps:
                dim = mul * ir_edge.dim
                self.edge_attr_slices.append((edge_offset, edge_offset + dim))
                edge_offset += dim

            self.num_l_blocks = len(self.edge_attr_slices)
            if self.edge_feats_irreps.num_irreps % self.num_l_blocks != 0:
                raise ValueError("edge_feats_irreps must split evenly across l blocks in radial v2")
            self.edge_feat_dim_per_l = self.edge_feats_irreps.num_irreps // self.num_l_blocks
            irreps_mid, instructions = tp_out_irreps_with_instructions(
                self.node_feats_irreps,
                self.edge_attrs_irreps,
                self.target_irreps,
            )
            self.conv_tp = o3.TensorProduct(
                self.node_feats_irreps,
                self.edge_attrs_irreps,
                irreps_mid,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
            )
            irreps_mid = irreps_mid.simplify()
            self.linear = o3.Linear(
                irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
            )

            self.v2_weight_ranges = [[] for _ in range(self.num_l_blocks)]
            offset = 0
            for ins in self.conv_tp.instructions:
                if not ins.has_weight:
                    continue
                width = int(np.prod(ins.path_shape))
                if width == 0:
                    continue
                self.v2_weight_ranges[ins.i_in2].append((offset, offset + width))
                offset += width
            if offset != self.conv_tp.weight_numel:
                raise ValueError("v2 weight layout does not match TensorProduct weight_numel")

            self.v2_block_weight_dims = []
            for idx in range(self.num_l_blocks):
                block_weight_dim = sum(end - start for start, end in self.v2_weight_ranges[idx])
                if block_weight_dim <= 0:
                    raise ValueError(f"No trainable TP weights found for l block {idx}")
                self.v2_block_weight_dims.append(block_weight_dim)
            self.v2_scalar_fastpath = False
            self.v2_in_mul = 0
            self.v2_out_mul_per_l = []
            self.v2_max_block_weight_dim = max(self.v2_block_weight_dims)
            self.conv_tp_weights_grouped = GroupedFullyConnectedNet(
                self.num_l_blocks,
                [self.edge_feat_dim_per_l] + self.radial_MLP + [self.v2_max_block_weight_dim],
            )

            if self.use_self_connection:
                self.skip_tp = o3.FullyConnectedTensorProduct(
                    self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
                )
            else:
                self.skip_tp = None
            self.reshape = reshape_irreps(self.irreps_out)
            return

        # get path for TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        #  MLP(TI) otimes Ylm
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # make weight with Rnl
        self.conv_tp_weights = nn.FullyConnectedNet(
            [self.edge_feats_irreps.num_irreps] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.Identity(),
        )

        # channel mix Linear
        irreps_mid = irreps_mid.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )
        # ZI otimes hI
        if self.use_self_connection:
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
            )
        else:
            self.skip_tp = None
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.linear1(node_feats)

        if self.radial_version == "v2":
            edge_feats_grouped = edge_feats.reshape(
                edge_feats.shape[0], self.num_l_blocks, self.edge_feat_dim_per_l
            )
            block_weights_all = self.conv_tp_weights_grouped(edge_feats_grouped)
            tp_weights = edge_feats.new_zeros(
                (edge_feats.shape[0], self.conv_tp.weight_numel)
            )
            for idx, _ in enumerate(self.edge_attr_slices):
                block_weights = block_weights_all[:, idx, : self.v2_block_weight_dims[idx]]
                block_offset = 0
                for weight_start, weight_end in self.v2_weight_ranges[idx]:
                    width = weight_end - weight_start
                    tp_weights[:, weight_start:weight_end] = block_weights[
                        :, block_offset : block_offset + width
                    ]
                    block_offset += width
            mji = self.conv_tp(
                node_feats[sender], edge_attrs, tp_weights
            )
            message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
            message = self.linear(message) / self.avg_num_neighbors
            sc = self.skip_tp(node_feats, node_attrs) if self.skip_tp is not None else None
            return (
                self.reshape(message),
                sc,
            )

        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        sc = self.skip_tp(node_feats, node_attrs) if self.skip_tp is not None else None
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]



@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(
        self,
        scale: float,
        shift: float,
        num_elements: Optional[int] = None,
        trainable_element_scales: bool = False,
    ):
        super().__init__()
        if trainable_element_scales:
            if num_elements is None:
                raise ValueError("num_elements is required for element-dependent scales")
            self.scale = None
            self.element_scales = torch.nn.Parameter(
                torch.ones(num_elements, dtype=torch.get_default_dtype())
            )
        else:
            self.register_buffer(
                "scale", torch.tensor(scale, dtype=torch.get_default_dtype())
            )
            self.element_scales = None
        self.register_buffer(
            "shift", torch.tensor(shift, dtype=torch.get_default_dtype())
        )

    def forward(
        self, x: torch.Tensor, node_attrs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        element_scales = getattr(self, "element_scales", None)
        if element_scales is None:
            return self.scale * x + self.shift
        if node_attrs is None:
            raise ValueError("node_attrs is required for element-dependent scales")
        scales = torch.matmul(
            node_attrs.to(dtype=element_scales.dtype), element_scales
        )
        return scales * x + self.shift

    def __repr__(self):
        element_scales = getattr(self, "element_scales", None)
        if element_scales is not None:
            scale_stats = (
                float(element_scales.min().item()),
                float(element_scales.max().item()),
            )
            return (
                f"{self.__class__.__name__}(element_scales=trainable[{scale_stats[0]:.6f},"
                f" {scale_stats[1]:.6f}], shift={self.shift:.6f})"
            )
        return f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
