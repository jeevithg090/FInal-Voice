"""Tests for scripts/prepare_metadata.py using small synthetic archives."""

from __future__ import annotations

import csv
import io
import wave
import zipfile
from pathlib import Path

import prepare_metadata as pm
import pytest

PHASE2 = "IndicTTS_Phase2_Tamil_fem_Speaker1_mono.zip"
PHASE3 = "IndicTTS_Phase3_Tamil_fem_Speaker1_mono.zip"


def make_wav(seconds: float, rate: int = 22050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def make_archive(path: Path, transcript_member: str, transcript: str, wavs: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(transcript_member, transcript.encode("utf-8"))
        for name, data in wavs.items():
            zf.writestr(name, data)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_archive(
        raw / PHASE2,
        "mono/Read_speech/txt.done.data",
        '﻿( p2_001 "வணக்கம்  உலகம்" )\n( p2_002 "a|b" )\nnot a prompt line\n( p2_003 "   " )\n',
        {
            "mono/Read_speech/wav/p2_001.wav": make_wav(1.0),
            "mono/Read_speech/wav/p2_002.wav": make_wav(0.5),
            "mono/Read_speech/wav/p2_003.wav": make_wav(0.5),  # empty text -> skipped
            "mono/Read_speech/wav/p2_999.wav": make_wav(0.5),  # no transcript -> skipped
            "mono/Additional_data/wav/add_1.wav": make_wav(0.5),
            "mono/Additional_data/all_bi_mono_sus_text.txt": b"",
        },
    )
    make_archive(
        raw / PHASE3,
        "speaker1/txt.done.data",
        '( s1_001 "தமிழ்" )\n',
        {"speaker1/wav/s1_001.wav": make_wav(2.0), "speaker1/readme.txt": b"ignored"},
    )
    return raw


class TestParsePrompts:
    def test_parses_festival_lines(self) -> None:
        text = '( utt_1 "hello world" )\n(utt_2 "second")\n'
        assert pm.parse_prompts(text) == {"utt_1": "hello world", "utt_2": "second"}

    def test_normalises_whitespace_and_pipes(self) -> None:
        assert pm.parse_prompts('( u "a |  b\tc" )') == {"u": "a b c"}

    def test_ignores_malformed_and_empty(self) -> None:
        text = 'garbage\n( missing_quotes )\n( u "" )\n( v " | " )\n\n'
        assert pm.parse_prompts(text) == {}


def test_normalize_text() -> None:
    assert pm.normalize_text("  x||y \n z ") == "x y z"


def test_build_jobs_additional_flag(tmp_path: Path) -> None:
    assert len(pm.build_jobs(tmp_path, PHASE2, PHASE3, include_additional=False)) == 2
    jobs = pm.build_jobs(tmp_path, PHASE2, PHASE3, include_additional=True)
    assert [j.out_subdir for j in jobs] == ["phase2", "phase3_s1", "phase2_additional"]


def test_process_jobs_pairs_audio_and_text(raw_dir: Path, tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    result = pm.process_jobs(pm.build_jobs(raw_dir, PHASE2, PHASE3, False), audio)

    assert result.rows == [
        ("phase2/p2_001.wav", "வணக்கம் உலகம்"),
        ("phase2/p2_002.wav", "a b"),
        ("phase3_s1/s1_001.wav", "தமிழ்"),
    ]
    assert result.skipped_without_text == 2
    assert result.total_seconds == pytest.approx(3.5)
    for rel, _ in result.rows:
        assert (audio / rel).is_file()
    assert not (audio / "phase2" / "p2_999.wav").exists()


def test_process_jobs_missing_archive(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pm.process_jobs(pm.build_jobs(tmp_path, PHASE2, PHASE3, False), tmp_path / "audio")


def test_process_jobs_missing_transcript(raw_dir: Path, tmp_path: Path) -> None:
    job = pm.Job(raw_dir / PHASE3, "speaker1/nope.data", "speaker1/wav/", "x")
    with pytest.raises(KeyError, match=r"nope\.data"):
        pm.process_jobs([job], tmp_path / "audio")


def test_write_metadata_round_trips(tmp_path: Path) -> None:
    rows = [("a/1.wav", 'text with "quotes"'), ("a/2.wav", "தமிழ்")]
    meta = tmp_path / "out" / "metadata.csv"
    pm.write_metadata(rows, meta)
    with meta.open(encoding="utf-8", newline="") as f:
        assert [tuple(r) for r in csv.reader(f, delimiter="|")] == rows


def test_main_success(raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    meta = tmp_path / "metadata.csv"
    argv = ["--raw-dir", str(raw_dir), "--audio-dir", str(tmp_path / "audio"), "--metadata", str(meta)]
    assert pm.main([*argv, "--min-rows", "3"]) == 0
    assert meta.read_text(encoding="utf-8").count("\n") == 3
    assert "rows: 3" in capsys.readouterr().out


def test_main_too_few_rows(raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    meta = tmp_path / "metadata.csv"
    argv = ["--raw-dir", str(raw_dir), "--audio-dir", str(tmp_path / "audio"), "--metadata", str(meta)]
    assert pm.main(argv) == 1
    assert not meta.exists()
    assert "Too few utterances" in capsys.readouterr().err


def test_main_missing_archive(tmp_path: Path) -> None:
    argv = ["--raw-dir", str(tmp_path), "--audio-dir", str(tmp_path / "a"), "--metadata", str(tmp_path / "m")]
    assert pm.main(argv) == 1
