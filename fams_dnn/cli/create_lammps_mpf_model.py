import sys

import torch
from e3nn.util import jit

from ptagnn.calculators import LAMMPS_MPF


def main():
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} model_path"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = sys.argv[1]  # takes model name as command-line input
    model = torch.load(model_path,map_location=device)
#    model = model.double().to("cpu")
    print(model)
    lammps_model = LAMMPS_MPF(model)
    lammps_model_compiled = jit.compile(lammps_model)
    lammps_model_compiled.save(model_path + "-lammps.pt")


if __name__ == "__main__":
    main()
