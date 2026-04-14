import logging
import random
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Set

import ase.io
import numpy as np
import torch
import torch.utils.data

from ptagnn.tools import AtomicNumberTable

from .atomic_data import AtomicData
from .utils import DEFAULT_CONFIG_TYPE_WEIGHTS, config_from_atoms


def _iter_xyz_atoms(file_path: str):
    yield from ase.io.iread(file_path, index=":")


def _is_isolated_atom(atoms) -> bool:
    return len(atoms) == 1 and atoms.info.get("config_type") == "IsolatedAtom"


@dataclass
class StreamingXYZStats:
    num_configs: int
    atomic_numbers: Set[int]
    atomic_energies_dict: Dict[int, float]
    has_stress_labels: bool
    has_virials_labels: bool


STREAM_CACHE_VERSION = 1


def default_stream_cache_path(cache_dir: str, file_path: str) -> str:
    dataset_path = os.path.realpath(file_path)
    digest = hashlib.sha1(dataset_path.encode("utf-8")).hexdigest()[:16]
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    stem = Path(file_path).name.replace(os.sep, "_")
    return str(cache_root / f"{stem}.{digest}.stream_cache.json")


def _normalized_config_type_weights(config_type_weights: Dict[str, float]) -> Dict[str, float]:
    return {
        str(key): float(value)
        for key, value in sorted(config_type_weights.items(), key=lambda item: str(item[0]))
    }


def _stream_signature(
    file_path: str,
    config_type_weights: Dict[str, float],
    energy_key: str,
    forces_key: str,
    stress_key: str,
    virials_key: str,
    dipole_key: str,
    charges_key: str,
) -> Dict[str, object]:
    stat_result = os.stat(file_path)
    return {
        "version": STREAM_CACHE_VERSION,
        "path": os.path.realpath(file_path),
        "size": int(stat_result.st_size),
        "mtime_ns": int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1e9))),
        "config_type_weights": _normalized_config_type_weights(config_type_weights),
        "energy_key": energy_key,
        "forces_key": forces_key,
        "stress_key": stress_key,
        "virials_key": virials_key,
        "dipole_key": dipole_key,
        "charges_key": charges_key,
    }


def _load_stream_cache(cache_path: str) -> Dict[str, object]:
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except FileNotFoundError:
        return {}
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Failed to read stream cache '%s': %s", cache_path, exc)
    return {}


def _write_stream_cache(cache_path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    os.replace(tmp_path, cache_path)


def _stream_stats_from_cache(payload: Dict[str, object]) -> StreamingXYZStats:
    return StreamingXYZStats(
        num_configs=int(payload["num_configs"]),
        atomic_numbers={int(z) for z in payload["atomic_numbers"]},
        atomic_energies_dict={
            int(z): float(value) for z, value in payload["atomic_energies_dict"].items()
        },
        has_stress_labels=bool(payload["has_stress_labels"]),
        has_virials_labels=bool(payload["has_virials_labels"]),
    )


def load_cached_stream_stats(
    cache_path: str,
    file_path: str,
    config_type_weights: Dict[str, float],
    energy_key: str,
    forces_key: str,
    stress_key: str,
    virials_key: str,
    dipole_key: str,
    charges_key: str,
) -> Optional[StreamingXYZStats]:
    cache = _load_stream_cache(cache_path)
    signature = _stream_signature(
        file_path=file_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
    )
    section = cache.get("scan")
    if isinstance(section, dict) and section.get("signature") == signature and isinstance(
        section.get("stats"), dict
    ):
        logging.info("Loaded streamed dataset metadata cache from '%s'", cache_path)
        return _stream_stats_from_cache(section["stats"])
    return None


def save_cached_stream_stats(
    cache_path: str,
    file_path: str,
    config_type_weights: Dict[str, float],
    energy_key: str,
    forces_key: str,
    stress_key: str,
    virials_key: str,
    dipole_key: str,
    charges_key: str,
    stats: StreamingXYZStats,
) -> None:
    cache = _load_stream_cache(cache_path)
    cache["scan"] = {
        "signature": _stream_signature(
            file_path=file_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
        ),
        "stats": {
            "num_configs": int(stats.num_configs),
            "atomic_numbers": sorted(int(z) for z in stats.atomic_numbers),
            "atomic_energies_dict": {
                str(int(z)): float(value) for z, value in stats.atomic_energies_dict.items()
            },
            "has_stress_labels": bool(stats.has_stress_labels),
            "has_virials_labels": bool(stats.has_virials_labels),
        },
    }
    _write_stream_cache(cache_path, cache)


def load_cached_average_e0s(
    cache_path: str,
    file_path: str,
    z_table: AtomicNumberTable,
    config_type_weights: Dict[str, float],
    energy_key: str,
    forces_key: str,
    stress_key: str,
    virials_key: str,
    dipole_key: str,
    charges_key: str,
) -> Optional[Dict[int, float]]:
    cache = _load_stream_cache(cache_path)
    signature = _stream_signature(
        file_path=file_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
    )
    section = cache.get("e0_average")
    if (
        isinstance(section, dict)
        and section.get("signature") == signature
        and section.get("zs") == [int(z) for z in z_table.zs]
        and isinstance(section.get("values"), dict)
    ):
        logging.info("Loaded streamed E0 cache from '%s'", cache_path)
        return {int(z): float(value) for z, value in section["values"].items()}
    return None


def save_cached_average_e0s(
    cache_path: str,
    file_path: str,
    z_table: AtomicNumberTable,
    config_type_weights: Dict[str, float],
    energy_key: str,
    forces_key: str,
    stress_key: str,
    virials_key: str,
    dipole_key: str,
    charges_key: str,
    e0s: Dict[int, float],
) -> None:
    cache = _load_stream_cache(cache_path)
    cache["e0_average"] = {
        "signature": _stream_signature(
            file_path=file_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
        ),
        "zs": [int(z) for z in z_table.zs],
        "values": {str(int(z)): float(value) for z, value in e0s.items()},
    }
    _write_stream_cache(cache_path, cache)


def scan_xyz_stream(
    file_path: str,
    config_type_weights: Optional[Dict[str, float]] = None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
    extract_atomic_energies: bool = False,
) -> StreamingXYZStats:
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    atomic_numbers: Set[int] = set()
    atomic_energies_dict: Dict[int, float] = {}
    has_stress_labels = False
    has_virials_labels = False
    num_configs = 0

    for idx, atoms in enumerate(_iter_xyz_atoms(file_path)):
        if extract_atomic_energies and _is_isolated_atom(atoms):
            if energy_key in atoms.info:
                atomic_energies_dict[int(atoms.get_atomic_numbers()[0])] = float(
                    atoms.info[energy_key]
                )
            else:
                logging.warning(
                    "Configuration '%d' is marked as 'IsolatedAtom' but does not contain an energy.",
                    idx,
                )
            continue

        config = config_from_atoms(
            atoms,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            config_type_weights=config_type_weights,
        )
        atomic_numbers.update(int(z) for z in config.atomic_numbers.tolist())
        has_stress_labels = has_stress_labels or float(config.stress_weight) > 0.0
        has_virials_labels = has_virials_labels or float(config.virials_weight) > 0.0
        num_configs += 1

    return StreamingXYZStats(
        num_configs=num_configs,
        atomic_numbers=atomic_numbers,
        atomic_energies_dict=atomic_energies_dict,
        has_stress_labels=has_stress_labels,
        has_virials_labels=has_virials_labels,
    )


def compute_average_E0s_from_stream(
    file_path: str,
    z_table: AtomicNumberTable,
    config_type_weights: Optional[Dict[str, float]] = None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    virials_key: str = "virials",
    dipole_key: str = "dipole",
    charges_key: str = "charges",
) -> Dict[int, float]:
    if config_type_weights is None:
        config_type_weights = DEFAULT_CONFIG_TYPE_WEIGHTS

    len_zs = len(z_table)
    ata = np.zeros((len_zs, len_zs))
    atb = np.zeros(len_zs)
    z_to_index = {int(z): idx for idx, z in enumerate(z_table.zs)}
    num_rows = 0

    for atoms in _iter_xyz_atoms(file_path):
        if _is_isolated_atom(atoms):
            continue

        config = config_from_atoms(
            atoms,
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            virials_key=virials_key,
            dipole_key=dipole_key,
            charges_key=charges_key,
            config_type_weights=config_type_weights,
        )
        counts = np.zeros(len_zs)
        for z in config.atomic_numbers:
            counts[z_to_index[int(z)]] += 1.0
        ata += np.outer(counts, counts)
        atb += counts * float(config.energy)
        num_rows += 1

    if num_rows == 0:
        logging.warning(
            "No non-isolated configurations found while computing streamed E0s; using zeros"
        )
        return {int(z): 0.0 for z in z_table.zs}

    try:
        e0s = np.linalg.lstsq(ata, atb, rcond=None)[0]
        return {int(z): float(e0s[idx]) for idx, z in enumerate(z_table.zs)}
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to compute streamed E0s using least squares regression, using zeros"
        )
        return {int(z): 0.0 for z in z_table.zs}


class StreamingAtomicDataDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_path: str,
        z_table: AtomicNumberTable,
        cutoff: float,
        config_type_weights: Optional[Dict[str, float]] = None,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        virials_key: str = "virials",
        dipole_key: str = "dipole",
        charges_key: str = "charges",
        num_configs: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 123,
        shuffle_buffer_size: int = 0,
        drop_remainder_across_ranks: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.z_table = z_table
        self.cutoff = cutoff
        self.config_type_weights = (
            config_type_weights
            if config_type_weights is not None
            else DEFAULT_CONFIG_TYPE_WEIGHTS
        )
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.virials_key = virials_key
        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.num_configs = num_configs
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.drop_remainder_across_ranks = drop_remainder_across_ranks
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rank_sample_limit(self) -> Optional[int]:
        if self.num_configs is None:
            return None
        if self.world_size <= 1:
            return self.num_configs
        if self.drop_remainder_across_ranks:
            return self.num_configs // self.world_size
        return (self.num_configs + self.world_size - 1) // self.world_size

    def __len__(self) -> int:
        rank_limit = self._rank_sample_limit()
        if rank_limit is None:
            raise TypeError("StreamingAtomicDataDataset length is unknown")
        return rank_limit

    def _iter_rank_configs(self) -> Iterator:
        rank_limit = self._rank_sample_limit()
        rank_seen = 0
        global_index = 0

        for atoms in _iter_xyz_atoms(self.file_path):
            if _is_isolated_atom(atoms):
                continue

            if self.world_size > 1 and global_index % self.world_size != self.rank:
                global_index += 1
                continue

            if rank_limit is not None and rank_seen >= rank_limit:
                break

            config = config_from_atoms(
                atoms,
                energy_key=self.energy_key,
                forces_key=self.forces_key,
                stress_key=self.stress_key,
                virials_key=self.virials_key,
                dipole_key=self.dipole_key,
                charges_key=self.charges_key,
                config_type_weights=self.config_type_weights,
            )
            rank_seen += 1
            global_index += 1
            yield config

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        rng = random.Random(
            self.seed + 1000003 * self.epoch + 8191 * self.rank + worker_id
        )

        def iter_worker_data():
            local_index = 0
            for config in self._iter_rank_configs():
                if local_index % num_workers == worker_id:
                    yield AtomicData.from_config(
                        config,
                        z_table=self.z_table,
                        cutoff=self.cutoff,
                    )
                local_index += 1

        if self.shuffle_buffer_size <= 1:
            yield from iter_worker_data()
            return

        buffer = []
        for item in iter_worker_data():
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(item)
                continue
            idx = rng.randrange(len(buffer))
            yield buffer[idx]
            buffer[idx] = item

        rng.shuffle(buffer)
        yield from buffer
