"""Deterministic parsers for the real-QEC records used in Run 6.

The Google archive stores detector samples as shot-aligned, little-endian
``.b8`` records in global Stim detector-declaration order.  Declaration order
is not a stable spatial order at the first and final detector rounds, so this
module derives the permutation from every ``DETECTOR(x, y, t)`` coordinate.

The PNNL/IBM release stores flattened syndrome measurements in JSON and
physical measurement assignments in OpenQASM 3.  QASM assignments, rather
than register-name suffixes, are authoritative for physical paths.

No function in this module selects an event window or decoder outcome.  Those
choices belong to the frozen Run 6 protocol.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_DETECTOR_RE = re.compile(r"^\s*DETECTOR\((?P<coords>[^)]*)\)")
_QASM_MEASUREMENT_RE = re.compile(
    r"^\s*(?P<register>c_(?:data|syndrome)_[A-Za-z0-9_]+)"
    r"\[(?P<index>\d+)\]\s*=\s*measure\s+\$(?P<qubit>\d+)\s*;"
)


@dataclass(frozen=True)
class DetectorLayout:
    """Coordinate-derived map from Stim declaration order to canonical roles."""

    declaration_coordinates: tuple[tuple[float, float, int], ...]
    roles: tuple[int, ...]
    canonical_checks: tuple[tuple[float, float], ...]
    ordered_declaration_indices: NDArray[np.int64]

    @property
    def detectors_per_shot(self) -> int:
        return len(self.declaration_coordinates)

    @property
    def checks_per_role(self) -> int:
        return len(self.canonical_checks)

    @property
    def role_count(self) -> int:
        return len(self.roles)

    @property
    def bytes_per_shot(self) -> int:
        return math.ceil(self.detectors_per_shot / 8)

    @property
    def padding_bits_per_shot(self) -> int:
        return 8 * self.bytes_per_shot - self.detectors_per_shot


@dataclass(frozen=True)
class QasmRegisterMap:
    """One logical register's measurement-to-physical-qubit assignments."""

    label: str
    data_qubits: tuple[int, ...]
    syndrome_qubits_by_round: tuple[tuple[int, ...], ...]

    @property
    def distance(self) -> int:
        return len(self.data_qubits)

    @property
    def rounds(self) -> int:
        return len(self.syndrome_qubits_by_round)

    @property
    def oriented_path(self) -> tuple[int, ...]:
        """Return ``data, check, data, ...`` for the first syndrome round."""

        if not self.syndrome_qubits_by_round:
            raise ValueError("At least one syndrome round is required.")
        checks = self.syndrome_qubits_by_round[0]
        if len(checks) != self.distance - 1:
            raise ValueError("A repetition-code path needs distance-1 checks.")
        path: list[int] = []
        for index, check in enumerate(checks):
            path.extend((self.data_qubits[index], check))
        path.append(self.data_qubits[-1])
        return tuple(path)


def _parse_integral_role(raw: str) -> int:
    value = float(raw.strip())
    rounded = round(value)
    if not np.isclose(value, rounded, atol=1e-12, rtol=0.0):
        raise ValueError(f"Detector time coordinate is not integral: {raw!r}.")
    return rounded


def parse_stim_detector_layout(
    circuit: str | Path,
    *,
    expected_roles: int | None = None,
    expected_checks_per_role: int | None = None,
) -> DetectorLayout:
    """Parse all explicit Stim detector coordinates and build a canonical map.

    The canonical check order is lexicographic in ``(x, y)``.  Every role must
    contain exactly the same set of unique checks.  The returned integer array
    has shape ``(role_count, checks_per_role)`` and contains indices into the
    one-dimensional global declaration order.
    """

    path = Path(circuit)
    coordinates: list[tuple[float, float, int]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = _DETECTOR_RE.match(line)
        if match is None:
            continue
        parts = [part.strip() for part in match.group("coords").split(",")]
        if len(parts) != 3:
            raise ValueError(f"{path}:{line_number}: Run 6 requires DETECTOR(x,y,t).")
        try:
            x = float(parts[0])
            y = float(parts[1])
            role = _parse_integral_role(parts[2])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid DETECTOR coordinates."
            ) from exc
        coordinates.append((x, y, role))
    if not coordinates:
        raise ValueError(f"No explicit DETECTOR(x,y,t) declarations in {path}.")

    roles = tuple(sorted({coordinate[2] for coordinate in coordinates}))
    if roles != tuple(range(roles[0], roles[0] + len(roles))):
        raise ValueError("Detector roles must be consecutive integers.")
    if expected_roles is not None:
        expected_role_values = tuple(range(expected_roles))
        if roles != expected_role_values:
            raise ValueError(
                f"Expected detector roles {expected_role_values}, found {roles}."
            )

    by_role: dict[int, dict[tuple[float, float], int]] = {}
    for declaration_index, (x, y, role) in enumerate(coordinates):
        key = (x, y)
        role_map = by_role.setdefault(role, {})
        if key in role_map:
            raise ValueError(f"Duplicate detector coordinate {key} in role {role}.")
        role_map[key] = declaration_index

    canonical_checks = tuple(sorted(by_role[roles[0]]))
    canonical_set = set(canonical_checks)
    if expected_checks_per_role is not None and (
        len(canonical_checks) != expected_checks_per_role
    ):
        raise ValueError(
            f"Expected {expected_checks_per_role} checks per role, "
            f"found {len(canonical_checks)}."
        )
    ordered_rows: list[list[int]] = []
    for role in roles:
        role_map = by_role[role]
        if set(role_map) != canonical_set:
            missing = sorted(canonical_set - set(role_map))
            extra = sorted(set(role_map) - canonical_set)
            raise ValueError(
                f"Role {role} has a different check set; "
                f"missing={missing}, extra={extra}."
            )
        ordered_rows.append([role_map[check] for check in canonical_checks])

    indices = np.asarray(ordered_rows, dtype=np.int64)
    indices.setflags(write=False)
    return DetectorLayout(
        declaration_coordinates=tuple(coordinates),
        roles=roles,
        canonical_checks=canonical_checks,
        ordered_declaration_indices=indices,
    )


def expected_b8_size_bytes(total_shots: int, layout: DetectorLayout) -> int:
    """Return the exact byte size for shot-aligned detector records."""

    if total_shots < 0:
        raise ValueError("total_shots must be nonnegative.")
    return int(total_shots) * layout.bytes_per_shot


def read_b8_detector_shots(
    path: str | Path,
    layout: DetectorLayout,
    *,
    start: int,
    stop: int,
    total_shots: int | None = None,
) -> NDArray[np.uint8]:
    """Read ``[start, stop)`` and return ``(shot, role, canonical_check)`` bits.

    Bits are unpacked little-endian within each byte.  Every shot begins at a
    byte boundary.  Any padding bits in the final byte of each shot must be
    zero; the Google Run 6 source has no padding because 1,224 is divisible by
    eight.
    """

    source = Path(path)
    if start < 0 or stop < start:
        raise ValueError("Require 0 <= start <= stop.")
    if total_shots is not None:
        expected = expected_b8_size_bytes(total_shots, layout)
        observed = source.stat().st_size
        if observed != expected:
            raise ValueError(
                f"Unexpected detector file size: expected {expected}, "
                f"observed {observed}."
            )
        if stop > total_shots:
            raise ValueError("Requested stop exceeds total_shots.")

    count = stop - start
    byte_count = count * layout.bytes_per_shot
    with source.open("rb") as handle:
        handle.seek(start * layout.bytes_per_shot)
        packed = handle.read(byte_count)
    if len(packed) != byte_count:
        raise ValueError(
            f"Short detector read: requested {byte_count} bytes, "
            f"received {len(packed)}."
        )
    if count == 0:
        return np.empty(
            (0, layout.role_count, layout.checks_per_role),
            dtype=np.uint8,
        )

    packed_matrix = np.frombuffer(packed, dtype=np.uint8).reshape(
        count,
        layout.bytes_per_shot,
    )
    unpacked = np.unpackbits(packed_matrix, axis=1, bitorder="little")
    if layout.padding_bits_per_shot:
        padding = unpacked[:, layout.detectors_per_shot :]
        if np.any(padding):
            raise ValueError("Nonzero per-shot padding bits in detector file.")
    declaration_bits = unpacked[:, : layout.detectors_per_shot]
    ordered = declaration_bits[:, layout.ordered_declaration_indices]
    return np.asarray(ordered, dtype=np.uint8)


def global_update_index(
    archive_shot: int,
    role_index: int,
    *,
    stream_start_shot: int,
    role_count: int,
) -> int:
    """Map shot-major replay coordinates to a zero-based update index."""

    local_shot = int(archive_shot) - int(stream_start_shot)
    if local_shot < 0:
        raise ValueError("archive_shot precedes stream_start_shot.")
    if not 0 <= role_index < role_count:
        raise ValueError("role_index is out of range.")
    return local_shot * role_count + int(role_index)


def update_coordinates(
    update_index: int,
    *,
    stream_start_shot: int,
    role_count: int,
) -> tuple[int, int]:
    """Invert :func:`global_update_index` for shot-major replay."""

    if update_index < 0 or role_count <= 0:
        raise ValueError("update_index must be nonnegative and role_count positive.")
    local_shot, role_index = divmod(int(update_index), int(role_count))
    return int(stream_start_shot) + local_shot, role_index


def flatten_shot_major(bits: ArrayLike) -> NDArray[np.uint8]:
    """Flatten ``(shot, role, check)`` in shot-major then role-major order."""

    array = np.asarray(bits, dtype=np.uint8)
    if array.ndim != 3:
        raise ValueError("bits must have shape (shot, role, check).")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("bits must be binary.")
    return array.reshape(array.shape[0] * array.shape[1], array.shape[2])


def parse_qasm_register_maps(
    path: str | Path,
    *,
    distance: int,
    rounds: int,
) -> dict[str, QasmRegisterMap]:
    """Parse state-specific OpenQASM measurement assignments.

    A returned dictionary key is the suffix after ``c_data_`` or
    ``c_syndrome_``.  Both register types must exist, data indices must cover
    ``0..distance-1``, syndrome indices must cover
    ``0..rounds*(distance-1)-1``, and the syndrome physical assignment must be
    stable across rounds.
    """

    if distance < 2 or rounds < 1:
        raise ValueError("distance >= 2 and rounds >= 1 are required.")
    assignments: dict[tuple[str, str], dict[int, int]] = {}
    source = Path(path)
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = _QASM_MEASUREMENT_RE.match(line)
        if match is None:
            continue
        register = match.group("register")
        if register.startswith("c_data_"):
            kind = "data"
            label = register[len("c_data_") :]
        elif register.startswith("c_syndrome_"):
            kind = "syndrome"
            label = register[len("c_syndrome_") :]
        else:  # pragma: no cover - excluded by the regular expression
            continue
        index = int(match.group("index"))
        qubit = int(match.group("qubit"))
        target = assignments.setdefault((label, kind), {})
        if index in target:
            raise ValueError(f"{source}:{line_number}: duplicate {register}[{index}].")
        target[index] = qubit

    labels = sorted({label for label, _ in assignments})
    result: dict[str, QasmRegisterMap] = {}
    for label in labels:
        data = assignments.get((label, "data"), {})
        syndrome = assignments.get((label, "syndrome"), {})
        expected_data = set(range(distance))
        expected_syndrome = set(range(rounds * (distance - 1)))
        if set(data) != expected_data or set(syndrome) != expected_syndrome:
            raise ValueError(
                f"Register {label!r} in {source} has incomplete measurements."
            )
        data_qubits = tuple(data[index] for index in range(distance))
        by_round = tuple(
            tuple(
                syndrome[round_index * (distance - 1) + check]
                for check in range(distance - 1)
            )
            for round_index in range(rounds)
        )
        if any(row != by_round[0] for row in by_round[1:]):
            raise ValueError(
                f"Register {label!r} changes syndrome qubits across rounds."
            )
        result[label] = QasmRegisterMap(
            label=label,
            data_qubits=data_qubits,
            syndrome_qubits_by_round=by_round,
        )
    if not result:
        raise ValueError(f"No complete repetition-code registers found in {source}.")
    return result


def repetition_detection_events(
    syndrome: ArrayLike,
    *,
    distance: int,
    rounds: int,
) -> NDArray[np.uint8]:
    """Return the paper's ``rounds × (distance-1)`` detection-event tensor.

    The first round is compared with an all-zero initial syndrome.  Every
    later round is XORed with its predecessor.  No terminal detector derived
    from final data measurements is appended.
    """

    flat = np.asarray(syndrome, dtype=np.uint8)
    if flat.ndim != 2 or flat.shape[1] != rounds * (distance - 1):
        raise ValueError("syndrome must have shape (shots, rounds*(distance-1)).")
    if np.any((flat != 0) & (flat != 1)):
        raise ValueError("syndrome values must be binary.")
    measured = flat.reshape(flat.shape[0], rounds, distance - 1)
    events = np.empty_like(measured)
    events[:, 0] = measured[:, 0]
    if rounds > 1:
        events[:, 1:] = measured[:, 1:] ^ measured[:, :-1]
    return events


def load_pnnl_register_arrays(
    bitstrings_path: str | Path,
    *,
    logical_state: int,
    register_label: str,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], dict[str, Any]]:
    """Load one state/register pair from a PNNL ``bitstrings.json`` file."""

    if logical_state not in (0, 1):
        raise ValueError("logical_state must be 0 or 1.")
    source = Path(bitstrings_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("PNNL bitstrings root must be a list.")
    matches = [
        entry
        for entry in payload
        if int(entry.get("metadata", {}).get("logical_state", -1)) == logical_state
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one logical-state {logical_state} entry, found {len(matches)}."
        )
    entry = matches[0]
    registers = entry.get("per_shot_cregs")
    if not isinstance(registers, dict):
        raise TypeError("per_shot_cregs must be an object.")
    data_key = f"c_data_{register_label}"
    syndrome_key = f"c_syndrome_{register_label}"
    if data_key not in registers or syndrome_key not in registers:
        raise KeyError(f"Missing {data_key!r} or {syndrome_key!r}.")
    data = np.asarray(registers[data_key], dtype=np.uint8)
    syndrome = np.asarray(registers[syndrome_key], dtype=np.uint8)
    if data.ndim != 2 or syndrome.ndim != 2 or data.shape[0] != syndrome.shape[0]:
        raise ValueError("PNNL register arrays have inconsistent shapes.")
    if np.any((data != 0) & (data != 1)) or np.any((syndrome != 0) & (syndrome != 1)):
        raise ValueError("PNNL register arrays must be binary.")
    return data, syndrome, dict(entry.get("metadata", {}))


def parse_dot01_outcomes(
    path: str | Path,
    *,
    expected_count: int | None = None,
) -> NDArray[np.uint8]:
    """Parse a newline-delimited Stim ``.01`` outcome file."""

    lines = Path(path).read_text(encoding="ascii").splitlines()
    if expected_count is not None and len(lines) != expected_count:
        raise ValueError(
            f"Expected {expected_count} outcome lines, found {len(lines)}."
        )
    if any(line not in {"0", "1"} for line in lines):
        raise ValueError("Outcome file contains values other than 0 or 1.")
    return np.fromiter((int(line) for line in lines), dtype=np.uint8, count=len(lines))


def require_disjoint_half_open_intervals(
    intervals: Iterable[tuple[str, int, int]],
) -> None:
    """Validate named half-open intervals without inspecting their contents."""

    ordered = sorted((name, int(start), int(stop)) for name, start, stop in intervals)
    for name, start, stop in ordered:
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid interval {name}: [{start}, {stop}).")
    by_start = sorted(ordered, key=lambda row: (row[1], row[2], row[0]))
    for left, right in pairwise(by_start):
        if right[1] < left[2]:
            raise ValueError(f"Intervals overlap: {left[0]} and {right[0]}.")
