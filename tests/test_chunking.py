from __future__ import annotations

from segmentwright_scribe.chunking import ChunkPlannerConfig, plan_chunks
from segmentwright_scribe.vad import VadSegment


def test_plan_chunks_prefers_vad_silence_gap_near_target() -> None:
    plan = plan_chunks(
        duration_sec=400.0,
        speech_segments=[
            VadSegment(0.0, 170.0),
            VadSegment(190.0, 370.0),
        ],
        source_id="source",
        config=ChunkPlannerConfig(target_chunk_sec=180.0, max_chunk_sec=220.0),
    )

    assert plan.split_times == [180.0]
    assert plan.forced_count == 0
    assert [(chunk.start_sec, chunk.end_sec) for chunk in plan.chunks] == [
        (0.0, 180.0),
        (180.0, 400.0),
    ]
    assert plan.chunks[1].cut_source == "vad_silence_gap"


def test_plan_chunks_uses_forced_fallback_when_no_silence_path_fits() -> None:
    plan = plan_chunks(
        duration_sec=500.0,
        speech_segments=[VadSegment(0.0, 500.0)],
        source_id="source",
        config=ChunkPlannerConfig(target_chunk_sec=180.0, max_chunk_sec=180.0),
    )

    assert plan.split_times == [180.0, 360.0]
    assert plan.forced_count == 2
    assert [(chunk.start_sample, chunk.end_sample) for chunk in plan.chunks] == [
        (0, 2_880_000),
        (2_880_000, 5_760_000),
        (5_760_000, 8_000_000),
    ]


def test_plan_chunks_merges_nearby_speech_segments() -> None:
    plan = plan_chunks(
        duration_sec=20.0,
        speech_segments=[
            VadSegment(1.0, 3.0, confidence=0.2),
            VadSegment(3.03, 4.0, confidence=0.7),
        ],
        source_id="source",
        config=ChunkPlannerConfig(max_chunk_sec=30.0),
    )

    assert plan.split_times == []
    assert len(plan.candidates) == 0
    assert plan.chunks[0].start_sec == 0.0
    assert plan.chunks[0].end_sec == 20.0
