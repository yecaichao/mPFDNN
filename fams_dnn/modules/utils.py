###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.utils.data
from scipy.constants import c, e

from ptagnn.tools import to_numpy
from ptagnn.tools.scatter import scatter_sum

from .blocks import AtomicEnergiesBlock


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def compute_edge_forces(
    energy: torch.Tensor, edge_vectors: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],
        inputs=[edge_vectors],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(edge_vectors)
    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.einsum(
            "zi,zi->z",
            cell[:, 0, :],
            torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
        ).unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def get_atomic_virials_stresses(
    edge_forces: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    receiver = edge_index[1]
    atomic_virials = scatter_sum(
        src=torch.einsum("zi,zj->zij", edge_forces, edge_vectors),
        index=receiver,
        dim=0,
        dim_size=batch.shape[0],
    )
    atomic_virials = 0.5 * (
        atomic_virials + atomic_virials.transpose(-1, -2)
    )
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs().unsqueeze(-1)
    atom_volume = volume[batch].view(-1, 1, 1)
    atomic_stress = torch.where(
        atom_volume > 0.0,
        -atomic_virials / atom_volume,
        torch.zeros_like(atomic_virials),
    )
    atomic_stress = torch.where(
        torch.abs(atomic_stress) < 1e10,
        atomic_stress,
        torch.zeros_like(atomic_stress),
    )
    return atomic_virials, atomic_stress


def compute_atomic_virials(
    energy: torch.Tensor,
    edge_vectors: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    training: bool = False,
) -> Optional[torch.Tensor]:
    edge_forces = compute_edge_forces(
        energy=energy, edge_vectors=edge_vectors, training=training
    )
    atomic_virials, _ = get_atomic_virials_stresses(
        edge_forces=edge_forces,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
        batch=batch,
        cell=cell,
    )
    return atomic_virials


def get_graph_virials_stress_from_atomic_virials(
    atomic_virials: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cell = cell.view(-1, 3, 3)
    num_graphs = cell.shape[0]
    graph_virials = scatter_sum(
        src=atomic_virials,
        index=batch,
        dim=0,
        dim_size=num_graphs,
    )
    volume = torch.linalg.det(cell).abs().view(-1, 1, 1)
    graph_stress = torch.where(
        volume > 0.0,
        -graph_virials / volume,
        torch.zeros_like(graph_virials),
    )
    graph_stress = torch.where(
        torch.abs(graph_stress) < 1e10,
        graph_stress,
        torch.zeros_like(graph_stress),
    )
    return graph_virials, graph_stress


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: Optional[torch.Tensor],
    cell: torch.Tensor,
    edge_vectors: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    batch: Optional[torch.Tensor] = None,
    atomic_virials: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    virials_impl: str = "atomic",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    use_atomic_virials = (
        virials_impl == "atomic"
        and (compute_virials or compute_stress)
        and batch is not None
        and (
            atomic_virials is not None
            or (edge_vectors is not None and edge_index is not None)
        )
    )
    if use_atomic_virials:
        if atomic_virials is None:
            assert edge_vectors is not None
            assert edge_index is not None
            atomic_virials = compute_atomic_virials(
                energy=energy,
                edge_vectors=edge_vectors,
                edge_index=edge_index,
                batch=batch,
                cell=cell,
                training=training,
            )
        virials, stress = get_graph_virials_stress_from_atomic_virials(
            atomic_virials=atomic_virials,
            batch=batch,
            cell=cell,
        )
        forces = (
            compute_forces(energy=energy, positions=positions, training=training)
            if compute_force
            else None
        )
    elif (compute_virials or compute_stress) and displacement is not None:
        # forces come for free
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(energy=energy, positions=positions, training=training),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    return forces, virials, stress


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def _check_non_zero(std):
    if std == 0.0:
        logging.warning(
            "Standard deviation of the scaling is zero, Changing to no scaling"
        )
        std = 1.0
    return std


def extract_invariant(x: torch.Tensor, num_layers: int, num_features: int, l_max: int):
    out = []
    for i in range(num_layers - 1):
        out.append(
            x[
                :,
                i
                * (l_max + 1) ** 2
                * num_features : (i * (l_max + 1) ** 2 + 1)
                * num_features,
            ]
        )
    out.append(x[:, -num_features:])
    return torch.cat(out, dim=-1)


def compute_mean_std_atomic_inter_energy(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()
    std = _check_non_zero(std)

    return mean, std


def compute_mean_rms_energy_forces(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    atom_energy_list = []
    forces_list = []

    for batch in data_loader:
        node_e0 = atomic_energies_fn(batch.node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs
        )
        graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
        atom_energy_list.append(
            (batch.energy - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

    atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
    forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

    mean = to_numpy(torch.mean(atom_energies)).item()
    rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()
    rms = _check_non_zero(rms)

    return mean, rms


def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
    num_neighbors = []

    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()

