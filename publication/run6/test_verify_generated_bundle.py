from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from analysis import extract_results
from analysis.test_extract_results import (
    FIXTURE_EVIDENCE_PATHS,
    make_fixture,
    run_fixture,
    write_json,
)
from verify_generated_bundle import (
    BundleVerificationError,
    _verify_bundle_with_validation_profile,
    verify_bundle,
)

PATH_KEY_BY_ROLE = {
    "detector_manifest": "detector",
    "freeze_ratification": "freeze_ratification",
    "repair_manifest": "repair_manifest",
    "repair_ratification": "repair_ratification",
    "randomization_manifest": "randomization",
    "pnnl_manifest": "pnnl",
    "pittsburgh_manifest": "pittsburgh",
    "outcome_manifest": "outcome",
}


def verify(paths: dict[str, Any], bundle: Path) -> None:
    _verify_bundle_with_validation_profile(
        detector_manifest=paths["detector"],
        freeze_ratification=paths["freeze_ratification"],
        repair_manifest=paths["repair_manifest"],
        repair_ratification=paths["repair_ratification"],
        randomization_manifest=paths["randomization"],
        pnnl_manifest=paths["pnnl"],
        pittsburgh_manifest=paths["pittsburgh"],
        outcome_manifest=paths["outcome"],
        bundle_dir=bundle,
        manuscript_source=Path(__file__).with_name("main.tex"),
        validation_profile=paths["_validation_profile"],
    )


def make_intermediate_symlink_attack(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path]:
    outside = tmp_path / "outside"
    source_paths = make_fixture(outside / "fixture")
    bundle = tmp_path / "generated"
    run_fixture(source_paths, bundle)

    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "fixture").symlink_to(
        outside / "fixture",
        target_is_directory=True,
    )
    attacked: dict[str, Any] = {
        PATH_KEY_BY_ROLE[role]: repository / relative
        for role, relative in FIXTURE_EVIDENCE_PATHS.items()
    }
    attacked["_validation_profile"] = replace(
        source_paths["_validation_profile"],
        repository_root=repository.resolve(),
    )
    return attacked, bundle


def test_production_entrypoints_expose_no_profile_override() -> None:
    assert set(inspect.signature(extract_results.main).parameters) == {"argv"}
    assert "_validation_profile" not in inspect.signature(verify_bundle).parameters
    assert "validation_profile" not in inspect.signature(verify_bundle).parameters


def test_canonical_evidence_path_accepts_makefile_dot_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    evidence = repository / "experiments" / "evidence.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}\n", encoding="utf-8")
    working = repository / "publication" / "run6"
    working.mkdir(parents=True)
    monkeypatch.chdir(working)

    observed = extract_results.require_canonical_repository_file(
        Path("../../experiments/evidence.json"),
        repository_root=repository,
        relative="experiments/evidence.json",
        context="test evidence",
    )
    assert observed == evidence


def test_fresh_extractor_bundle_is_accepted(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated"
    run_fixture(paths, bundle)
    verify(paths, bundle)


def test_manual_claim_edit_is_rejected(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated"
    run_fixture(paths, bundle)
    (bundle / "claim_sentence.tex").write_text(
        "Unsupported manual result.\\n",
        encoding="utf-8",
    )
    with pytest.raises(
        BundleVerificationError,
        match="not extractor-reproducible",
    ):
        verify(paths, bundle)


def test_symlinked_bundle_root_is_rejected(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated-real"
    run_fixture(paths, bundle)
    link = tmp_path / "generated-link"
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(
        BundleVerificationError,
        match="Publication bundle has a symlink component",
    ):
        verify(paths, link)


def test_symlinked_manuscript_source_is_rejected(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated"
    run_fixture(paths, bundle)
    manuscript = tmp_path / "main-real.tex"
    manuscript.write_text(
        Path(__file__).with_name("main.tex").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    link = tmp_path / "main-link.tex"
    link.symlink_to(manuscript)
    with pytest.raises(
        BundleVerificationError,
        match="Manuscript source has a symlink component",
    ):
        _verify_bundle_with_validation_profile(
            detector_manifest=paths["detector"],
            freeze_ratification=paths["freeze_ratification"],
            repair_manifest=paths["repair_manifest"],
            repair_ratification=paths["repair_ratification"],
            randomization_manifest=paths["randomization"],
            pnnl_manifest=paths["pnnl"],
            pittsburgh_manifest=paths["pittsburgh"],
            outcome_manifest=paths["outcome"],
            bundle_dir=bundle,
            manuscript_source=link,
            validation_profile=paths["_validation_profile"],
        )


def test_evidence_path_edit_is_rejected(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated"
    run_fixture(paths, bundle)
    manifest_path = bundle / "publication_bundle_manifest.json"
    manifest = extract_results.load_json(manifest_path)
    manifest["evidence_inputs"]["detector_manifest"]["path"] = (
        "fixture/fabricated/detector_freeze_manifest.json"
    )
    write_json(manifest_path, manifest)
    with pytest.raises(
        BundleVerificationError,
        match="not extractor-reproducible",
    ):
        verify(paths, bundle)


def test_extractor_rejects_intermediate_evidence_symlink(tmp_path: Path) -> None:
    paths, _ = make_intermediate_symlink_attack(tmp_path)
    with pytest.raises(
        ValueError,
        match="symlink component",
    ):
        run_fixture(paths, tmp_path / "attacked-output")


def test_verifier_rejects_intermediate_evidence_symlink(tmp_path: Path) -> None:
    paths, bundle = make_intermediate_symlink_attack(tmp_path)
    with pytest.raises(
        ValueError,
        match="symlink component",
    ):
        verify(paths, bundle)


def test_missing_verified_macro_consumption_is_rejected(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated"
    run_fixture(paths, bundle)
    manuscript = tmp_path / "main.tex"
    source = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")
    manuscript.write_text(
        source.replace(r"\RunSixVerifiedGateTable", "", 1),
        encoding="utf-8",
    )
    with pytest.raises(
        BundleVerificationError,
        match="consumed exactly once",
    ):
        _verify_bundle_with_validation_profile(
            detector_manifest=paths["detector"],
            freeze_ratification=paths["freeze_ratification"],
            repair_manifest=paths["repair_manifest"],
            repair_ratification=paths["repair_ratification"],
            randomization_manifest=paths["randomization"],
            pnnl_manifest=paths["pnnl"],
            pittsburgh_manifest=paths["pittsburgh"],
            outcome_manifest=paths["outcome"],
            bundle_dir=bundle,
            manuscript_source=manuscript,
            validation_profile=paths["_validation_profile"],
        )


def test_verified_claim_moved_outside_result_branch_is_rejected(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated"
    run_fixture(paths, bundle)
    manuscript = tmp_path / "main.tex"
    source = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")
    source = source.replace(
        r"\newcommand{\RunSixClaimSentence}{%",
        "\\RunSixVerifiedClaim\n" r"\newcommand{\RunSixClaimSentence}{%",
        1,
    ).replace(
        r"    \RunSixVerifiedClaim%",
        "    %",
        1,
    )
    manuscript.write_text(source, encoding="utf-8")
    with pytest.raises(
        BundleVerificationError,
        match="consumed exactly once",
    ):
        _verify_bundle_with_validation_profile(
            detector_manifest=paths["detector"],
            freeze_ratification=paths["freeze_ratification"],
            repair_manifest=paths["repair_manifest"],
            repair_ratification=paths["repair_ratification"],
            randomization_manifest=paths["randomization"],
            pnnl_manifest=paths["pnnl"],
            pittsburgh_manifest=paths["pittsburgh"],
            outcome_manifest=paths["outcome"],
            bundle_dir=bundle,
            manuscript_source=manuscript,
            validation_profile=paths["_validation_profile"],
        )


def test_result_conditional_without_local_else_is_rejected(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path / "input")
    bundle = tmp_path / "generated"
    run_fixture(paths, bundle)
    manuscript = tmp_path / "main.tex"
    source = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")
    source = source.replace(
        (
            "\\ifrunresultbundle\n"
            "  \\input{generated/manuscript_artifact_contract.tex}\n"
            "\\else\n"
            "\\fi"
        ),
        (
            "\\ifrunresultbundle\n"
            "  \\input{generated/manuscript_artifact_contract.tex}\n"
            "\\fi"
        ),
        1,
    )
    manuscript.write_text(source, encoding="utf-8")
    with pytest.raises(
        BundleVerificationError,
        match="local else/fi pair",
    ):
        _verify_bundle_with_validation_profile(
            detector_manifest=paths["detector"],
            freeze_ratification=paths["freeze_ratification"],
            repair_manifest=paths["repair_manifest"],
            repair_ratification=paths["repair_ratification"],
            randomization_manifest=paths["randomization"],
            pnnl_manifest=paths["pnnl"],
            pittsburgh_manifest=paths["pittsburgh"],
            outcome_manifest=paths["outcome"],
            bundle_dir=bundle,
            manuscript_source=manuscript,
            validation_profile=paths["_validation_profile"],
        )
