#!/usr/bin/env python3
# ==============================================================================#
#
#  Regime-Gated Residual Mixture-of-Experts for Cross-Sectional Volatility Forecasting
#
#  Deterministic generator and auditor for the pinned dependency manifest.
#
#  Author: CS Chirinda
#  License: MIT
#  Version: 1.0.0
#
# ==============================================================================#

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    packages_distributions,
)
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Deque, Dict, List, Literal, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequirementsManifest:
    """Immutable audit record describing one synthesised dependency manifest."""

    manifest_path: Path
    manifest_text: str
    manifest_sha256: str
    stdlib_modules: Tuple[str, ...]
    direct_pins: Tuple[Tuple[str, str], ...]
    transitive_pins: Tuple[Tuple[str, str], ...]
    unresolved_imports: Tuple[str, ...]
    interpreter: str = field(default_factory=lambda: sys.version.split()[0])
    generated_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def synthesize_pinned_requirements_manifest(
    direct_imports: Sequence[str],
    output_path: Path,
    *,
    optional_imports: Optional[Dict[str, str]] = None,
    annotations: Optional[Dict[str, str]] = None,
    banner_title: str = (
        "Regime-Gated Residual Mixture-of-Experts for Cross-Sectional "
        "Volatility Forecasting"
    ),
    pin_operator: Literal["==", "~=", ">="] = "==",
    include_transitive: bool = True,
    write_sidecar_audit: bool = True,
    dry_run: bool = False,
) -> RequirementsManifest:
    """Synthesise a fully pinned, banner-headed ``requirements.txt`` from a live env.

    The routine partitions an observed import surface into standard-library
    modules (never pinned) and distribution-backed modules (always pinned),
    resolves import names to their true distribution names via
    :func:`importlib.metadata.packages_distributions` — closing the
    ``sklearn`` -> ``scikit-learn`` trap that silently installs a deprecation
    stub — and then performs a breadth-first traversal of the installed
    requirement graph to pin the complete transitive closure.

    Pinning the closure rather than only the direct roots is the operative
    reproducibility guarantee for this study. A 30-seed replication compared
    under a Diebold-Mariano test with Newey-West(4) standard errors is only
    valid if seed ``s`` on host A executes the identical numerical stack as
    seed ``s`` on host B; an unpinned transitive such as ``threadpoolctl``
    (BLAS thread count) or ``narwhals`` (dataframe dispatch) perturbs pooled
    loss aggregates below the significance thresholds the study reports.

    Marker handling is deliberately dependency-free: requirements gated behind
    an ``extra ==`` marker are excluded by regex, while all other environment
    markers (``python_version``, ``sys_platform``) are resolved *empirically* —
    a marker-excluded distribution is simply not installed and is therefore
    pruned by :class:`PackageNotFoundError`. This avoids importing
    ``packaging`` merely to evaluate markers, which would add a node to the
    very graph being pinned.

    Args:
        direct_imports: Top-level import names observed in the target module
            header, e.g. ``("numpy", "pandas", "torch", "scipy", "sklearn",
            "matplotlib")``. Standard-library entries may be included freely;
            they are detected and excluded from pinning.
        output_path: Destination path for the rendered manifest. Parent
            directories are created if absent.
        optional_imports: Mapping of import name to pin specifier for
            dependencies guarded by ``try/except ImportError`` in the target
            module (e.g. ``{"yfinance": "yfinance==1.6.0"}``). Emitted
            commented-out under a dedicated section, because the modelling and
            evaluation stack must install in air-gapped environments.
        annotations: Optional mapping of distribution name to a one-line
            rationale comment emitted immediately above that pin.
        banner_title: Title rendered inside the ``# ===...===#`` header block,
            matching the house style of the reference implementation.
        pin_operator: Version specifier operator. ``"=="`` is the only
            defensible choice for replication; ``"~="`` and ``">="`` permit
            resolver drift and are provided solely for downstream library
            packaging.
        include_transitive: When ``True``, pin the full breadth-first closure.
            Disabling this yields a manifest incompatible with
            ``pip install --require-hashes``, which is all-or-nothing.
        write_sidecar_audit: When ``True``, emit ``<output>.audit.json``
            carrying the SHA-256 digest, interpreter version, and pin tables
            for the experiment provenance registry.
        dry_run: When ``True``, render and hash the manifest but perform no
            filesystem writes.

    Returns:
        RequirementsManifest: Frozen audit record containing the rendered text,
        its SHA-256 digest, the stdlib exclusion list, the direct and
        transitive pin tables, and any import names that resolved to no
        installed distribution.

    Raises:
        ValueError: If ``direct_imports`` is empty, or if every supplied name
            resolves to the standard library, leaving nothing to pin.

    Example:
        >>> record = synthesize_pinned_requirements_manifest(
        ...     direct_imports=("numpy", "pandas", "torch", "scipy",
        ...                     "sklearn", "matplotlib", "json", "hashlib"),
        ...     output_path=Path("requirements.txt"),
        ...     optional_imports={"yfinance": "yfinance==1.6.0"},
        ...     dry_run=True,
        ... )
        >>> len(record.direct_pins)
        6
    """
    # Reject an empty import surface outright: an empty manifest is a silent
    # reproducibility failure rather than a benign no-op.
    if not direct_imports:
        raise ValueError("direct_imports must contain at least one module name.")

    # Normalise the optional and annotation mappings so downstream lookups
    # never need a None guard.
    optional_imports = dict(optional_imports or {})
    annotations = dict(annotations or {})

    # Snapshot the interpreter's authoritative stdlib module set; this is the
    # only correct source of truth and it varies by CPython minor version.
    stdlib_names: Set[str] = set(sys.stdlib_module_names)

    # Build the import-name -> distribution-name index once, since it walks
    # every installed distribution's top-level metadata and is not cheap.
    dist_index: Dict[str, List[str]] = packages_distributions()

    # PEP 503 normaliser: fold case and collapse runs of '-', '_', '.' so that
    # 'typing_extensions' and 'typing-extensions' compare equal.
    def _normalise(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    # Accumulate the stdlib modules encountered, purely for the audit record.
    stdlib_hits: List[str] = []
    # Accumulate the distribution names backing the direct imports.
    root_distributions: List[str] = []
    # Accumulate import names that resolved to no installed distribution.
    unresolved: List[str] = []

    # Classify every supplied import name exactly once, preserving input order.
    for module_name in direct_imports:
        # Reduce dotted paths ('importlib.metadata') to their top-level root,
        # because only the root can be attributed to a distribution.
        root_module = module_name.split(".")[0]
        # Divert standard-library modules to the exclusion list; pinning them
        # is a hazard that shadows the interpreter with a stale backport.
        if root_module in stdlib_names:
            stdlib_hits.append(module_name)
            continue
        # Resolve the import name to its owning distribution(s). This is the
        # step that maps 'sklearn' to 'scikit-learn'.
        owners = dist_index.get(root_module, [])
        # Record an unresolved import rather than guessing a distribution name;
        # a fabricated pin is worse than a declared gap.
        if not owners:
            unresolved.append(module_name)
            continue
        # Register every owning distribution, de-duplicating on the PEP 503
        # normalised form while preserving the metadata's canonical spelling.
        for owner in owners:
            if _normalise(owner) not in {_normalise(d) for d in root_distributions}:
                root_distributions.append(owner)

    # Refuse to emit a manifest with no third-party pins at all.
    if not root_distributions:
        raise ValueError(
            "No installed third-party distributions resolved from direct_imports; "
            f"unresolved={unresolved!r}"
        )

    # Regex capturing the leading distribution name of a requirement string,
    # discarding version specifiers, extras brackets, and environment markers.
    requirement_head = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

    # Seed the breadth-first frontier with the direct roots.
    frontier: Deque[str] = deque(root_distributions)
    # Track visited distributions by normalised name to prevent cycles.
    visited: Set[str] = {_normalise(name) for name in root_distributions}
    # Preserve discovery order so the emitted manifest is byte-stable.
    discovered: List[str] = []

    # Traverse the installed requirement graph until the frontier is exhausted.
    while frontier:
        # Pop in FIFO order so roots are emitted before their dependencies.
        current = frontier.popleft()
        try:
            # Load the installed distribution's metadata; absence means the
            # node was excluded by an environment marker and must be pruned.
            current_dist = distribution(current)
        except PackageNotFoundError:
            continue
        # Record the resolved node for later version pinning.
        discovered.append(current)
        # Stop expanding once transitives are disabled, but keep the roots.
        if not include_transitive:
            continue
        # Walk this node's declared requirements.
        for raw_requirement in current_dist.requires or []:
            # Drop requirements gated behind an unrequested extra; installing
            # them would inflate the closure with test and docs tooling.
            if "extra ==" in raw_requirement:
                continue
            # Extract the bare distribution name from the requirement string.
            head = requirement_head.match(raw_requirement)
            if head is None:
                continue
            child = head.group(1)
            # Enqueue unseen children, marking them visited immediately so a
            # diamond dependency is expanded exactly once.
            if _normalise(child) not in visited:
                visited.add(_normalise(child))
                frontier.append(child)

    # Resolve a concrete installed version for every discovered node.
    resolved_pins: List[Tuple[str, str]] = [
        (name, dist_version(name)) for name in discovered
    ]
    # Partition the pin table into direct roots and transitive dependencies.
    root_keys = {_normalise(name) for name in root_distributions}
    direct_pins = tuple(p for p in resolved_pins if _normalise(p[0]) in root_keys)
    transitive_pins = tuple(
        sorted(
            (p for p in resolved_pins if _normalise(p[0]) not in root_keys),
            key=lambda pair: pair[0].lower(),
        )
    )

    # Fixed-width rule used to draw the banner and section separators.
    rule = "# " + "=" * 78 + "#"
    # Open the banner block in the reference implementation's house style.
    lines: List[str] = [rule, "#", f"#  {banner_title}", "#"]
    # State the manifest's purpose and its reproducibility contract.
    lines += [
        "#  Pinned runtime dependency manifest. The complete transitive closure is",
        f"#  pinned with '{pin_operator}' so that seed s on host A runs the identical",
        "#  numerical stack as seed s on host B, preserving the validity of paired",
        "#  Diebold-Mariano comparisons with Newey-West(4) standard errors.",
        "#",
        f"#  Interpreter        : CPython {sys.version.split()[0]} on {sys.platform}",
        f"#  Direct pins        : {len(direct_pins)}",
        f"#  Transitive pins    : {len(transitive_pins)}",
        f"#  Stdlib (unpinned)  : {len(stdlib_hits)}",
        "#  Generated (UTC)    : "
        + datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "#",
        "#  Install:  python -m pip install --require-virtualenv -r requirements.txt",
        "#  Verify :  python -m pip check",
        "#",
        rule,
        "",
    ]
    # Document the deliberate stdlib exclusions so the omission reads as a
    # decision rather than an oversight during audit.
    if stdlib_hits:
        lines += [
            "# Standard library (deliberately NOT pinned; shipped with CPython):",
            "#   " + ", ".join(sorted(stdlib_hits)),
            "",
        ]
    # Surface any import that resolved to no installed distribution, so the
    # gap is visible instead of silently absent.
    if unresolved:
        lines += [
            "# UNRESOLVED — no installed distribution provides these imports:",
            "#   " + ", ".join(sorted(unresolved)),
            "",
        ]

    # Emit the direct-dependency section.
    lines += ["# --- Direct dependencies " + "-" * 54, ""]
    for name, ver in direct_pins:
        # Prepend the caller-supplied rationale when one exists.
        note = annotations.get(name) or annotations.get(_normalise(name))
        if note:
            lines.append(f"# {note}")
        # Emit the pin itself using the configured operator.
        lines.append(f"{name}{pin_operator}{ver}")
        lines.append("")

    # Emit the transitive closure section when transitives are included.
    if include_transitive and transitive_pins:
        lines += [
            "# --- Pinned transitive closure " + "-" * 48,
            "# Required in full for 'pip install --require-hashes', which is",
            "# all-or-nothing: every requirement must carry a hash.",
            "",
        ]
        # Column-align the pins for legibility during manual review.
        width = max(len(f"{n}{pin_operator}{v}") for n, v in transitive_pins) + 2
        for name, ver in transitive_pins:
            spec = f"{name}{pin_operator}{ver}"
            note = annotations.get(name) or annotations.get(_normalise(name))
            lines.append(f"{spec.ljust(width)}# {note}" if note else spec)
        lines.append("")

    # Emit guarded optional dependencies commented-out, mirroring the target
    # module's try/except ImportError contract for air-gapped execution.
    if optional_imports:
        lines += [
            "# --- Optional acquisition extras " + "-" * 46,
            "# Guarded by try/except ImportError in the target module: the modelling",
            "# and evaluation stack must install and run without network access.",
            "# Install only on the ingestion host that materialises the raw panel.",
            "",
        ]
        for import_name, specifier in sorted(optional_imports.items()):
            lines.append(f"# {import_name}:")
            lines.append(f"# {specifier}")
        lines.append("")

    # Close the manifest with a terminal rule.
    lines += [rule, "#                              END OF MANIFEST", rule, ""]
    # Join with LF only, so the digest is identical on POSIX and Windows.
    manifest_text = "\n".join(lines)
    # Hash the exact bytes that will be written, forming the provenance key.
    manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    # Materialise the manifest unless the caller requested a dry run.
    if not dry_run:
        # Create the parent directory tree if the destination is nested.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write with an explicit encoding and newline policy so the on-disk
        # bytes match the hashed bytes exactly on every platform.
        output_path.write_text(manifest_text, encoding="utf-8", newline="\n")
        # Record the write and its digest in the audit trail.
        logger.info(
            "Wrote %s (%d direct, %d transitive pins) sha256=%s",
            output_path,
            len(direct_pins),
            len(transitive_pins),
            manifest_sha256,
        )
        # Emit the machine-readable sidecar for the provenance registry.
        if write_sidecar_audit:
            sidecar = output_path.with_suffix(output_path.suffix + ".audit.json")
            sidecar.write_text(
                json.dumps(
                    {
                        "manifest_sha256": manifest_sha256,
                        "interpreter": sys.version.split()[0],
                        "platform": sys.platform,
                        "generated_utc": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                        "direct_pins": dict(direct_pins),
                        "transitive_pins": dict(transitive_pins),
                        "stdlib_excluded": sorted(stdlib_hits),
                        "unresolved_imports": sorted(unresolved),
                        "optional_extras": optional_imports,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    else:
        # Announce the suppressed write so a dry run is never mistaken for one.
        logger.info(
            "Dry run: manifest rendered but not written (sha256=%s)",
            manifest_sha256,
        )

    # Return the frozen audit record for downstream provenance capture.
    return RequirementsManifest(
        manifest_path=output_path,
        manifest_text=manifest_text,
        manifest_sha256=manifest_sha256,
        stdlib_modules=tuple(sorted(stdlib_hits)),
        direct_pins=direct_pins,
        transitive_pins=transitive_pins,
        unresolved_imports=tuple(sorted(unresolved)),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    record = synthesize_pinned_requirements_manifest(
        direct_imports=(
            "re", "hashlib", "inspect", "itertools", "json", "logging", "math",
            "random", "sys", "dataclasses", "datetime", "pathlib", "typing",
            "matplotlib", "numpy", "pandas", "torch", "scipy", "sklearn",
        ),
        output_path=Path("requirements.generated.txt"),
        optional_imports={
            "yfinance": "yfinance==1.6.0",
            "pandas_market_calendars": "pandas-market-calendars==5.4.0",
        },
        annotations={
            "numpy": "Panel arrays, rolling kernels, 30-seed RNG registry.",
            "torch": "GELU block, frozen base, K residual experts, softmax gate.",
            "scikit-learn": "Ridge for the pooled HAR baseline.",
            "threadpoolctl": "Pin BLAS threads to 1 for byte-stable loss aggregates.",
        },
        dry_run=True,
    )
    print(record.manifest_text)
    print(f"sha256={record.manifest_sha256}")
