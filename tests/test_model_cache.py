from dataclasses import replace

from segmentwright_scribe.cache import TranscriptCache, TranscriptCacheKey


def test_previous_model_cache_is_not_reused(tmp_path):
    cache = TranscriptCache(tmp_path)
    old = TranscriptCacheKey("source", 0.0, 1.0, "audiohash", "microsoft/mai-transcribe-1.5", "ja")
    new = replace(old, model="microsoft/mai-transcribe-2")
    cache.save(old, text="old transcript", metadata={})
    assert cache.load(new) is None
    cache.save(new, text="new transcript", metadata={})
    assert cache.load(old).text == "old transcript"
    assert cache.load(new).text == "new transcript"
