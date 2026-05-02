#!/usr/bin/env python3
"""Stream rover LMDB experience records into msgpack-gz shards for training.

Use case: the Jetson rover writes :class:`MouseDroidExperienceRecord`
instances into an LMDB store via :class:`mousedroid.experience.logger`. The
training cluster needs them as portable msgpack-gz shards (one per
``--shard-size`` records) to feed into the offline RL pipeline.

This CLI reads the LMDB through the *Phase 2* :class:`LMDBReplayReader`
async chunked stream (memory-bounded — never loads the whole DB into
RAM, matching the 8 GB Jetson Orin Nano budget). Each chunk is bundled
into a shard, msgpack-encoded, gzipped, and written to ``--dest``.

Usage::

    # Smoke run, no side effects:
    python scripts/export_experience_to_training.py \\
        --lmdb /home/jetson/mousedroid_experience \\
        --dest /tmp/shards --dry-run

    # Real export (writes shards):
    python scripts/export_experience_to_training.py \\
        --lmdb /home/jetson/mousedroid_experience \\
        --dest /var/training/shards \\
        --shard-size 1024

The CLI emits structured-log lines for every transition so an operator
can tail the rover journal:

- ``export_started`` (path, dest, shard_size, dry_run)
- ``export_chunk_decoded`` (n=<chunk len>, total=<so far>)
- ``export_shard_written`` (path, n_records, n_bytes)
- ``export_dry_run_complete`` / ``export_complete`` (totals)

Exit codes:
  0  success
  2  CLI usage error
  3  source LMDB missing or unreadable
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import sys
import time
from pathlib import Path

import msgpack

# When invoked as a script (``python scripts/export_experience_to_training.py``)
# the package isn't on sys.path. Mirror the trick the rest of scripts/* uses.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mousedroid.config.schema import ExperienceConfig  # noqa: E402
from mousedroid.experience.record import MouseDroidExperienceRecord  # noqa: E402
from mousedroid.logging.setup import get_logger  # noqa: E402
from mousedroid.training.replay.lmdb_reader import LMDBReplayReader  # noqa: E402

_log = get_logger(__name__)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_LMDB_ERR = 3

# Defaults that respect the 8 GB Orin Nano budget. shard_size at 256
# records ≈ a few MB per shard with the default record geometry.
DEFAULT_SHARD_SIZE: int = 256
DEFAULT_CHUNK_SIZE: int = 64
DEFAULT_MAP_SIZE_GB: float = 40.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
    )
    parser.add_argument(
        "--lmdb",
        type=Path,
        required=True,
        help="Source LMDB env directory (e.g. /home/jetson/mousedroid_experience).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory for msgpack-gz shards. Created if missing.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=f"Records per shard (default {DEFAULT_SHARD_SIZE}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            f"LMDB read chunk size (default {DEFAULT_CHUNK_SIZE}); "
            "memory-bounded — see LMDBReplayReader docs."
        ),
    )
    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=DEFAULT_MAP_SIZE_GB,
        help=(
            "Source LMDB map size hint in GB. "
            f"Default {DEFAULT_MAP_SIZE_GB} matches `experience.map_size_gb` "
            "in `config/jetson_production.yaml`."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read records but write no shards. Useful for size estimation + sanity.",
    )
    return parser.parse_args(argv)


def _record_to_dict(record: MouseDroidExperienceRecord) -> dict[str, object]:
    """Serialise one record into a plain dict for msgpack encoding.

    Mirrors :meth:`MouseDroidExperienceRecord.serialize` but keeps the
    payload as a python dict so the shard can hold many records without
    re-decoding each one.
    """
    return {
        "schema_version": record.schema_version,
        "timestamp": record.timestamp,
        "vision_features": record.vision_features.tobytes(),
        "vision_features_shape": list(record.vision_features.shape),
        "distance_m": record.distance_m,
        "motor_state": record.motor_state.tobytes(),
        "motor_state_shape": list(record.motor_state.shape),
        "action": record.action.tobytes(),
        "action_shape": list(record.action.shape),
        "reward": record.reward,
        "surprise": record.surprise,
    }


def _write_shard(
    dest_dir: Path,
    shard_index: int,
    records: list[MouseDroidExperienceRecord],
) -> Path:
    """Write one msgpack-gzipped shard. Returns the destination path."""
    payload = {
        "schema_version": (
            records[0].schema_version if records else MouseDroidExperienceRecord().schema_version
        ),
        "n_records": len(records),
        "records": [_record_to_dict(r) for r in records],
    }
    body = msgpack.packb(payload)
    shard_path = dest_dir / f"shard-{shard_index:06d}.msgpack.gz"
    with gzip.open(shard_path, "wb") as fh:
        fh.write(body)
    return shard_path


async def _drive(args: argparse.Namespace) -> int:
    """Async driver — pulls chunks, batches into shards, writes them out."""
    lmdb_path: Path = args.lmdb
    dest: Path = args.dest

    if not lmdb_path.exists():
        _log.error("export_lmdb_missing", path=str(lmdb_path))
        return EXIT_LMDB_ERR

    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    experience_cfg = ExperienceConfig(
        path=str(lmdb_path),
        map_size_gb=args.map_size_gb,
        flush_every_n=1,
        export_path=str(dest),
    )
    reader = LMDBReplayReader(experience_cfg)

    _log.info(
        "export_started",
        lmdb=str(lmdb_path),
        dest=str(dest),
        shard_size=args.shard_size,
        chunk_size=args.chunk_size,
        dry_run=args.dry_run,
    )

    started = time.monotonic()
    buf: list[MouseDroidExperienceRecord] = []
    total_records = 0
    shards_written = 0

    async for chunk in reader.stream(args.chunk_size):
        total_records += len(chunk)
        _log.debug(
            "export_chunk_decoded",
            n=len(chunk),
            total=total_records,
        )
        buf.extend(chunk)
        while len(buf) >= args.shard_size:
            shard, buf = buf[: args.shard_size], buf[args.shard_size :]
            if not args.dry_run:
                path = _write_shard(dest, shards_written, shard)
                _log.info(
                    "export_shard_written",
                    path=str(path),
                    n_records=len(shard),
                    n_bytes=path.stat().st_size,
                )
            shards_written += 1

    # Tail shard (smaller than shard-size).
    if buf:
        if not args.dry_run:
            path = _write_shard(dest, shards_written, buf)
            _log.info(
                "export_shard_written",
                path=str(path),
                n_records=len(buf),
                n_bytes=path.stat().st_size,
                tail=True,
            )
        shards_written += 1

    elapsed_s = time.monotonic() - started
    summary = {
        "total_records": total_records,
        "shards": shards_written,
        "elapsed_s": round(elapsed_s, 3),
        "skipped_schema_mismatch": reader.stats["skipped_schema_mismatch"],
    }
    if args.dry_run:
        _log.info("export_dry_run_complete", **summary)
    else:
        _log.info("export_complete", dest=str(dest), **summary)

    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — wraps :func:`_drive` in `asyncio.run`."""
    args = _parse_args(argv)
    if args.shard_size <= 0:
        _log.error("export_invalid_shard_size", shard_size=args.shard_size)
        return EXIT_USAGE
    if args.chunk_size <= 0:
        _log.error("export_invalid_chunk_size", chunk_size=args.chunk_size)
        return EXIT_USAGE
    return asyncio.run(_drive(args))


if __name__ == "__main__":
    raise SystemExit(main())
