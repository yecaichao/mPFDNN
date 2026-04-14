from typing import Dict, List, Optional

import torch
from e3nn.util.jit import compile_mode

from ptagnn.tools.scatter import scatter_sum


@compile_mode("script")
class LAMMPS_MPF(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
        compute_atomic_virials: bool = False,
        atomic_virials_chunk_size: int = 16,
        virials_impl: str = "atomic",
    ) -> Dict[str, Optional[torch.Tensor]]:
        num_graphs = data["ptr"].numel() - 1
        compute_displacement = compute_virials and virials_impl == "displacement"
        request_atomic_virials = compute_atomic_virials or (
            compute_virials and virials_impl == "atomic"
        )
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=compute_displacement,
            compute_atomic_virials=request_atomic_virials,
            atomic_virials_chunk_size=atomic_virials_chunk_size,
            virials_impl=virials_impl,
        )
        node_energy = out["node_energy"]
        if node_energy is None:
            return {
                "total_energy_local": None,
                "node_energy": None,
                "forces": None,
                "virials": None,
                "atomic_virials": None,
            }
        positions = data["positions"]
        displacement = out["displacement"]
        atomic_virials = out.get("atomic_virials", None)
        forces: Optional[torch.Tensor] = torch.zeros_like(positions)
        virials: Optional[torch.Tensor] = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        # accumulate energies of local atoms
        node_energy_local = node_energy * local_or_ghost
        if atomic_virials is not None:
            atomic_virials = atomic_virials * local_or_ghost.view(-1, 1, 1)
        total_energy_local = scatter_sum(
            src=node_energy_local, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        # compute partial forces and (possibly) partial virials
        grad_outputs: List[Optional[torch.Tensor]] = [
            torch.ones_like(total_energy_local)
        ]
        if compute_virials and virials_impl == "displacement" and displacement is not None:
            forces, virials = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions, displacement],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if virials is not None:
                virials = -1 * virials
            else:
                virials = torch.zeros_like(displacement)
        else:
            forces = torch.autograd.grad(
                outputs=[total_energy_local],
                inputs=[positions],
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if forces is not None:
                forces = -1 * forces
            else:
                forces = torch.zeros_like(positions)
            if compute_virials and atomic_virials is not None:
                virials = scatter_sum(
                    src=atomic_virials,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
            "atomic_virials": atomic_virials,
        }
