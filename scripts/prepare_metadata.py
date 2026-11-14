#!/usr/bin/env python3
"""Build a Piper-compatible single-speaker dataset from IndicTTS Tamil archives.

Reads the IITM IndicTTS Phase 2 / Phase 3 Tamil female mono ZIP archives,
pairs each WAV file with its transcript from the Festival-style
``txt.done.data`` file, copies the audio into ``--audio-dir`` and writes a
``metadata.csv`` (``relative/path.wav|text``) that ``python -m piper.train``
consumes.

Only the Python standard library is used so the script can run before any
training dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import wave
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Festival prompt line: ( utt_id "transcript text" )
PROMPT_LINE = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)\s*$')
WHITESPACE = re.compile(r"\s+")

DEFAULT_MIN_ROWS = 3000


@dataclass(frozen=True)
class Job:
    """One transcript file + WAV directory inside an archive."""

    archive: Path
    transcript_member: str
    wav_prefix: str
    out_subdir: str


@dataclass(frozen=True)
class Result:
    rows: list[tuple[str, str]]
    total_seconds: float
    skipped_without_text: int


def normalize_text(text: str) -> str:
    """Collapse whitespace and drop the CSV field delimiter used by Piper."""
    return WHITESPACE.sub(" ", text.replace("|", " ")).strip()


def parse_prompts(text: str) -> dict[str, str]:
    """Parse Festival ``txt.done.data`` content into ``{utt_id: text}``.

    Malformed lines and entries whose text is empty after normalisation are
    ignored.
    """
    prompts: dict[str, str] = {}
    for raw_line in text.splitlines():
        match = PROMPT_LINE.match(raw_line.strip())
        if not match:
            continue
        utt_id, transcript = match.group(1), normalize_text(match.group(2))
        if transcript:
            prompts[utt_id] = transcript
    return prompts


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        return wav.getnframes() / float(rate) if rate else 0.0


def build_jobs(raw_dir: Path, phase2_zip: str, phase3_zip: str, include_additional: bool) -> list[Job]:
    phase2 = raw_dir / phase2_zip
    phase3 = raw_dir / phase3_zip
    jobs = [
        Job(phase2, "mono/Read_speech/txt.done.data", "mono/Read_speech/wav/", "phase2"),
        Job(phase3, "speaker1/txt.done.data", "speaker1/wav/", "phase3_s1"),
    ]
    if include_additional:
        jobs.append(
            Job(
                phase2,
                "mono/Additional_data/all_bi_mono_sus_text.txt",
                "mono/Additional_data/wav/",
                "phase2_additional",
            )
        )
    return jobs


def process_jobs(jobs: Iterable[Job], audio_dir: Path) -> Result:
    """Extract paired audio for every job and return the metadata rows."""
    rows: list[tuple[str, str]] = []
    total_seconds = 0.0
    skipped = 0

    for job in jobs:
        if not job.archive.is_file():
            raise FileNotFoundError(f"Missing archive: {job.archive}")
        with zipfile.ZipFile(job.archive) as zf:
            try:
                transcript = zf.read(job.transcript_member)
            except KeyError as exc:
                raise KeyError(f"{job.transcript_member!r} not found in {job.archive.name}") from exc
            prompts = parse_prompts(transcript.decode("utf-8-sig", errors="ignore"))

            out_dir = audio_dir / job.out_subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            for member in zf.namelist():
                if not (member.startswith(job.wav_prefix) and member.lower().endswith(".wav")):
                    continue
                stem = Path(member).stem
                text = prompts.get(stem)
                if not text:
                    skipped += 1
                    continue

                dst = out_dir / f"{stem}.wav"
                with zf.open(member) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out)
                total_seconds += wav_duration_seconds(dst)
                rows.append((dst.relative_to(audio_dir).as_posix(), text))

    rows.sort(key=lambda row: row[0])
    return Result(rows=rows, total_seconds=total_seconds, skipped_without_text=skipped)


def write_metadata(rows: Sequence[tuple[str, str]], meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL).writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory holding the phase ZIPs")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Output directory for WAV files")
    parser.add_argument("--metadata", type=Path, required=True, help="Output metadata.csv path")
    parser.add_argument("--phase2-zip", default="IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip")
    parser.add_argument("--phase3-zip", default="IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip")
    parser.add_argument(
        "--include-additional", action="store_true", help="Also use Phase 2 Additional_data utterances"
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help="Fail if fewer utterances than this are found (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = build_jobs(args.raw_dir, args.phase2_zip, args.phase3_zip, args.include_additional)
    try:
        result = process_jobs(jobs, args.audio_dir)
    except (FileNotFoundError, KeyError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if len(result.rows) < args.min_rows:
        print(
            f"ERROR: Too few utterances: {len(result.rows)} (expected at least {args.min_rows})",
            file=sys.stderr,
        )
        return 1

    write_metadata(result.rows, args.metadata)
    print(f"metadata: {args.metadata}")
    print(f"rows: {len(result.rows)}")
    print(f"skipped (no transcript): {result.skipped_without_text}")
    print(f"hours: {result.total_seconds / 3600:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
