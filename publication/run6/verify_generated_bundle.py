"""Recompute and byte-verify the hash-bound Run 6 publication bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path

from analysis import extract_results


class BundleVerificationError(RuntimeError):
    """Raised when generated manuscript artifacts are not reproducible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_non_symlink_path(
    path: Path,
    *,
    kind: str,
    context: str,
) -> Path:
    """Return an absolute lexical path after rejecting every symlink component."""

    lexical = Path(os.path.abspath(path))
    for component in reversed((lexical, *lexical.parents)):
        if component.is_symlink():
            raise BundleVerificationError(
                f"{context} has a symlink component: {component}"
            )
    if kind == "directory":
        valid = lexical.is_dir()
    elif kind == "file":
        valid = lexical.is_file()
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"Unsupported path kind: {kind}")
    if not valid:
        raise BundleVerificationError(
            f"{context} must be an existing regular {kind}: {lexical}"
        )
    return lexical


def flat_file_map(root: Path) -> dict[str, Path]:
    root = require_non_symlink_path(
        root,
        kind="directory",
        context="Publication bundle",
    )
    result: dict[str, Path] = {}
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise BundleVerificationError(
                f"Bundle entries must be regular, non-symlink files: {path}"
            )
        result[path.name] = path
    return result


def strip_tex_comments(source: str) -> str:
    """Remove unescaped TeX comments before static macro accounting."""

    retained = []
    for line in source.splitlines():
        limit = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                limit = index
                break
        retained.append(line[:limit])
    return "\n".join(retained)


def verify_manuscript_macro_consumption(
    *, contract_path: Path, manuscript_source: Path
) -> None:
    """Require every verified include macro exactly once in a result branch."""

    manuscript_source = require_non_symlink_path(
        manuscript_source,
        kind="file",
        context="Manuscript source",
    )
    contract = strip_tex_comments(contract_path.read_text(encoding="utf-8"))
    declaration_pattern = re.compile(
        r"\\newcommand\s*\{\s*\\(RunSixVerified[A-Za-z]+)\s*\}"
    )
    declarations = declaration_pattern.findall(contract)
    declared = set(declarations)
    if not declarations or len(declarations) != len(declared):
        raise BundleVerificationError(
            "Verified manuscript contract has missing or duplicate macro declarations."
        )

    manuscript = strip_tex_comments(manuscript_source.read_text(encoding="utf-8"))
    use_pattern = re.compile(r"\\(RunSixVerified[A-Za-z]+)")
    all_uses = use_pattern.findall(manuscript)
    unknown = sorted(set(all_uses) - declared)
    if unknown:
        raise BundleVerificationError(
            f"Manuscript consumes undeclared verified macros: {unknown}"
        )
    result_branch_pattern = re.compile(
        r"(?<!\\newif)\\ifrunresultbundle(?P<true>.*?)\\else(?P<false>.*?)\\fi",
        re.DOTALL,
    )
    result_branch_starts = re.findall(
        r"(?<!\\newif)\\ifrunresultbundle",
        manuscript,
    )
    result_branch_matches = list(result_branch_pattern.finditer(manuscript))
    if len(result_branch_matches) != len(result_branch_starts):
        raise BundleVerificationError(
            "Every result conditional must have a local else/fi pair and "
            "result conditionals cannot be nested."
        )
    result_branch_source = "\n".join(
        match.group("true") for match in result_branch_matches
    )
    result_uses = use_pattern.findall(result_branch_source)
    invalid = {
        name: {
            "whole_source": all_uses.count(name),
            "result_branch": result_uses.count(name),
        }
        for name in sorted(declared)
        if all_uses.count(name) != 1 or result_uses.count(name) != 1
    }
    if invalid:
        raise BundleVerificationError(
            "Every RunSixVerified macro must be consumed exactly once in a "
            f"result-branch source block; invalid={invalid}"
        )


def _verify_bundle_with_validation_profile(
    *,
    detector_manifest: Path,
    freeze_ratification: Path,
    repair_manifest: Path,
    repair_ratification: Path,
    randomization_manifest: Path,
    pnnl_manifest: Path,
    pittsburgh_manifest: Path,
    outcome_manifest: Path,
    bundle_dir: Path,
    manuscript_source: Path,
    validation_profile: extract_results.ValidationProfile,
) -> None:
    profile = validation_profile
    source_paths = {
        "detector_manifest": detector_manifest,
        "freeze_ratification": freeze_ratification,
        "repair_manifest": repair_manifest,
        "repair_ratification": repair_ratification,
        "randomization_manifest": randomization_manifest,
        "pnnl_manifest": pnnl_manifest,
        "pittsburgh_manifest": pittsburgh_manifest,
        "outcome_manifest": outcome_manifest,
    }
    expected_evidence = extract_results.evidence_input_records(
        source_paths,
        profile=profile,
    )
    _, _, expected_publication_provenance = (
        extract_results.validate_publication_repair_provenance(
            profile=profile,
            evidence_records=expected_evidence,
        )
    )
    sources = {role: path.resolve(strict=True) for role, path in source_paths.items()}
    if len(set(sources.values())) != 8:
        raise BundleVerificationError("The eight evidence inputs must be distinct.")

    with tempfile.TemporaryDirectory(prefix="run6-publication-verify-") as temporary:
        expected_dir = Path(temporary) / "generated"
        status = extract_results._run_with_validation_profile(
            [
                "--detector-manifest",
                str(sources["detector_manifest"]),
                "--freeze-ratification",
                str(sources["freeze_ratification"]),
                "--repair-manifest",
                str(sources["repair_manifest"]),
                "--repair-ratification",
                str(sources["repair_ratification"]),
                "--randomization-manifest",
                str(sources["randomization_manifest"]),
                "--pnnl-manifest",
                str(sources["pnnl_manifest"]),
                "--pittsburgh-manifest",
                str(sources["pittsburgh_manifest"]),
                "--outcome-manifest",
                str(sources["outcome_manifest"]),
                "--output-dir",
                str(expected_dir),
            ],
            validation_profile=profile,
        )
        if status != 0:
            raise BundleVerificationError(
                f"Publication extractor returned nonzero status {status}."
            )

        expected = flat_file_map(expected_dir)
        observed = flat_file_map(bundle_dir)
        if set(observed) != set(expected):
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            raise BundleVerificationError(
                f"Generated bundle contract changed; missing={missing}, extra={extra}"
            )
        for name in sorted(expected):
            expected_path = expected[name]
            observed_path = observed[name]
            if (
                observed_path.stat().st_size != expected_path.stat().st_size
                or sha256_file(observed_path) != sha256_file(expected_path)
            ):
                raise BundleVerificationError(
                    f"Generated artifact is not extractor-reproducible: {name}"
                )

    required = {
        "publication_bundle_manifest.json",
        "manuscript_artifact_contract.json",
        "manuscript_artifact_contract.tex",
        "claim_sentence.tex",
    }
    if not required <= set(observed):
        raise BundleVerificationError(
            "Bundle lacks its manifest, manuscript contract, or recomputed claim."
        )
    manifest = extract_results.load_json(observed["publication_bundle_manifest.json"])
    if manifest.get("schema_version") != "run6-publication-bundle-v6":
        raise BundleVerificationError(
            "Publication bundle is not the required three-layer-provenance v6 schema."
        )
    if manifest.get("evidence_inputs") != expected_evidence:
        raise BundleVerificationError(
            "Publication bundle evidence roles, paths, or hashes changed."
        )
    if manifest.get("publication_provenance_inputs") != expected_publication_provenance:
        raise BundleVerificationError(
            "Publication-repair provenance paths or hashes changed."
        )
    verify_manuscript_macro_consumption(
        contract_path=observed["manuscript_artifact_contract.tex"],
        manuscript_source=manuscript_source,
    )


def verify_bundle(
    *,
    detector_manifest: Path,
    freeze_ratification: Path,
    repair_manifest: Path,
    repair_ratification: Path,
    randomization_manifest: Path,
    pnnl_manifest: Path,
    pittsburgh_manifest: Path,
    outcome_manifest: Path,
    bundle_dir: Path,
    manuscript_source: Path,
) -> None:
    """Production verifier with immutable Git anchors and no profile override."""

    _verify_bundle_with_validation_profile(
        detector_manifest=detector_manifest,
        freeze_ratification=freeze_ratification,
        repair_manifest=repair_manifest,
        repair_ratification=repair_ratification,
        randomization_manifest=randomization_manifest,
        pnnl_manifest=pnnl_manifest,
        pittsburgh_manifest=pittsburgh_manifest,
        outcome_manifest=outcome_manifest,
        bundle_dir=bundle_dir,
        manuscript_source=manuscript_source,
        validation_profile=extract_results.load_production_validation_profile(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--freeze-ratification", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--repair-ratification", type=Path, required=True)
    parser.add_argument("--randomization-manifest", type=Path, required=True)
    parser.add_argument("--pnnl-manifest", type=Path, required=True)
    parser.add_argument("--pittsburgh-manifest", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, default=Path("main.tex"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_bundle(
        detector_manifest=args.detector_manifest,
        freeze_ratification=args.freeze_ratification,
        repair_manifest=args.repair_manifest,
        repair_ratification=args.repair_ratification,
        randomization_manifest=args.randomization_manifest,
        pnnl_manifest=args.pnnl_manifest,
        pittsburgh_manifest=args.pittsburgh_manifest,
        outcome_manifest=args.outcome_manifest,
        bundle_dir=args.bundle_dir,
        manuscript_source=args.manuscript,
    )
    print("Run 6 publication bundle exactly matches a fresh extractor run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
