"""Synthetic parser tests that do not inspect any held Run 6 values."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from aoc.qec_real import (
    expected_b8_size_bytes,
    flatten_shot_major,
    global_update_index,
    load_pnnl_register_arrays,
    parse_dot01_outcomes,
    parse_qasm_register_maps,
    parse_stim_detector_layout,
    read_b8_detector_shots,
    repetition_detection_events,
    update_coordinates,
)


def _pack_little_endian(records: np.ndarray) -> bytes:
    return np.packbits(records, axis=1, bitorder="little").tobytes()


def test_stim_coordinates_correct_boundary_declaration_reversal(
    tmp_path: Path,
) -> None:
    circuit = tmp_path / "tiny.stim"
    circuit.write_text(
        """DETECTOR(1, 1, 0) rec[-1]
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(1, 0, 1) rec[-3]
DETECTOR(1, 1, 1) rec[-4]
""",
        encoding="utf-8",
    )
    layout = parse_stim_detector_layout(
        circuit,
        expected_roles=2,
        expected_checks_per_role=2,
    )
    assert layout.canonical_checks == ((1.0, 0.0), (1.0, 1.0))
    assert layout.ordered_declaration_indices.tolist() == [[1, 0], [2, 3]]


def test_expected_stim_roles_must_start_at_zero(tmp_path: Path) -> None:
    circuit = tmp_path / "shifted.stim"
    circuit.write_text(
        """DETECTOR(0, 0, 1) rec[-1]
DETECTOR(0, 0, 2) rec[-2]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected detector roles"):
        parse_stim_detector_layout(
            circuit,
            expected_roles=2,
            expected_checks_per_role=1,
        )


def test_b8_is_shot_aligned_little_endian_and_coordinate_reordered(
    tmp_path: Path,
) -> None:
    circuit = tmp_path / "eight.stim"
    circuit.write_text(
        """DETECTOR(0, 1, 0) rec[-1]
DETECTOR(0, 0, 0) rec[-2]
DETECTOR(1, 1, 0) rec[-3]
DETECTOR(1, 0, 0) rec[-4]
DETECTOR(0, 0, 1) rec[-5]
DETECTOR(0, 1, 1) rec[-6]
DETECTOR(1, 0, 1) rec[-7]
DETECTOR(1, 1, 1) rec[-8]
""",
        encoding="utf-8",
    )
    layout = parse_stim_detector_layout(circuit)
    declarations = np.array(
        [
            [1, 0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    source = tmp_path / "events.b8"
    source.write_bytes(_pack_little_endian(declarations))
    assert expected_b8_size_bytes(2, layout) == 2

    observed = read_b8_detector_shots(
        source,
        layout,
        start=0,
        stop=2,
        total_shots=2,
    )
    expected = declarations[:, layout.ordered_declaration_indices]
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(
        flatten_shot_major(observed),
        expected.reshape(4, 4),
    )


def test_b8_rejects_nonzero_padding_and_wrong_file_size(tmp_path: Path) -> None:
    circuit = tmp_path / "three.stim"
    circuit.write_text(
        """DETECTOR(0, 0, 0) rec[-1]
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(2, 0, 0) rec[-3]
""",
        encoding="utf-8",
    )
    layout = parse_stim_detector_layout(circuit)
    bad_padding = tmp_path / "bad-padding.b8"
    bad_padding.write_bytes(bytes([0b10000101]))
    with pytest.raises(ValueError, match="padding"):
        read_b8_detector_shots(
            bad_padding,
            layout,
            start=0,
            stop=1,
            total_shots=1,
        )
    with pytest.raises(ValueError, match="file size"):
        read_b8_detector_shots(
            bad_padding,
            layout,
            start=0,
            stop=1,
            total_shots=2,
        )


def test_update_index_round_trip_is_shot_major() -> None:
    assert (
        global_update_index(
            40_003,
            7,
            stream_start_shot=40_000,
            role_count=51,
        )
        == 160
    )
    assert update_coordinates(
        160,
        stream_start_shot=40_000,
        role_count=51,
    ) == (40_003, 7)


def test_qasm_mapping_is_authoritative_and_requires_stable_checks(
    tmp_path: Path,
) -> None:
    qasm = tmp_path / "circuit.qasm"
    qasm.write_text(
        """OPENQASM 3.0;
bit[3] c_data_misleading_suffix;
bit[4] c_syndrome_misleading_suffix;
c_syndrome_misleading_suffix[0] = measure $11;
c_syndrome_misleading_suffix[1] = measure $13;
c_syndrome_misleading_suffix[2] = measure $11;
c_syndrome_misleading_suffix[3] = measure $13;
c_data_misleading_suffix[0] = measure $10;
c_data_misleading_suffix[1] = measure $12;
c_data_misleading_suffix[2] = measure $14;
""",
        encoding="utf-8",
    )
    mapping = parse_qasm_register_maps(qasm, distance=3, rounds=2)
    register = mapping["misleading_suffix"]
    assert register.data_qubits == (10, 12, 14)
    assert register.syndrome_qubits_by_round == ((11, 13), (11, 13))
    assert register.oriented_path == (10, 11, 12, 13, 14)


def test_repetition_events_have_no_terminal_detector() -> None:
    syndrome = np.array(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    observed = repetition_detection_events(syndrome, distance=3, rounds=3)
    expected = np.array(
        [
            [[1, 0], [0, 1], [1, 0]],
            [[0, 0], [0, 1], [1, 0]],
        ],
        dtype=np.uint8,
    )
    assert observed.shape == (2, 3, 2)
    np.testing.assert_array_equal(observed, expected)


def test_pnnl_loader_selects_logical_state_by_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bitstrings.json"
    path.write_text(
        """
[
  {
    "metadata": {"logical_state": 1, "basis": "Z"},
    "per_shot_cregs": {
      "c_data_path": [[1, 0, 1]],
      "c_syndrome_path": [[1, 0, 0, 1]]
    }
  },
  {
    "metadata": {"logical_state": 0, "basis": "Z"},
    "per_shot_cregs": {
      "c_data_path": [[0, 0, 0]],
      "c_syndrome_path": [[0, 1, 1, 0]]
    }
  }
]
""".strip(),
        encoding="utf-8",
    )
    data, syndrome, metadata = load_pnnl_register_arrays(
        path,
        logical_state=0,
        register_label="path",
    )
    np.testing.assert_array_equal(data, [[0, 0, 0]])
    np.testing.assert_array_equal(syndrome, [[0, 1, 1, 0]])
    assert metadata["logical_state"] == 0


def test_dot01_parser_validates_line_count_and_alphabet(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.01"
    path.write_text("0\n1\n0\n", encoding="ascii")
    np.testing.assert_array_equal(
        parse_dot01_outcomes(path, expected_count=3),
        [0, 1, 0],
    )
    with pytest.raises(ValueError, match="Expected"):
        parse_dot01_outcomes(path, expected_count=2)
