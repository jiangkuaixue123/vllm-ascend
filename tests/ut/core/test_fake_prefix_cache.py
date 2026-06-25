from types import SimpleNamespace

import pytest

from vllm_ascend.core.fake_prefix_cache import maybe_apply_fake_prefix_cache


def make_request(num_prompt_tokens: int, num_tokens: int | None = None):
    return SimpleNamespace(
        num_prompt_tokens=num_prompt_tokens,
        num_tokens=num_prompt_tokens if num_tokens is None else num_tokens,
    )


def test_fake_prefix_cache_disabled(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO", raising=False)

    result = maybe_apply_fake_prefix_cache(
        make_request(100),
        block_size=16,
        num_local_computed_tokens=0,
        num_external_computed_tokens=0,
        load_kv_async=False,
    )

    assert result == 0


def test_fake_prefix_cache_adds_external_tokens(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO", "0.9")

    result = maybe_apply_fake_prefix_cache(
        make_request(100),
        block_size=16,
        num_local_computed_tokens=32,
        num_external_computed_tokens=16,
        load_kv_async=False,
    )

    assert result == 80 - 32


def test_fake_prefix_cache_uses_block_alignment(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO", "0.9")

    result = maybe_apply_fake_prefix_cache(
        make_request(100),
        block_size=16,
        num_local_computed_tokens=0,
        num_external_computed_tokens=0,
        load_kv_async=False,
    )

    assert result == 80


def test_fake_prefix_cache_keeps_last_token_uncached(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO", "1")

    result = maybe_apply_fake_prefix_cache(
        make_request(32),
        block_size=16,
        num_local_computed_tokens=0,
        num_external_computed_tokens=0,
        load_kv_async=False,
    )

    assert result == 16


def test_fake_prefix_cache_skips_async_kv_load(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO", "0.9")

    result = maybe_apply_fake_prefix_cache(
        make_request(100),
        block_size=16,
        num_local_computed_tokens=0,
        num_external_computed_tokens=16,
        load_kv_async=True,
    )

    assert result == 16


def test_fake_prefix_cache_rejects_invalid_ratio(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO", "1.1")

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        maybe_apply_fake_prefix_cache(
            make_request(100),
            block_size=16,
            num_local_computed_tokens=0,
            num_external_computed_tokens=0,
            load_kv_async=False,
        )


def test_fake_prefix_cache_rejects_negative_ratio(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FAKE_PREFIX_CACHE_HIT_RATIO", "-0.1")

    with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
        maybe_apply_fake_prefix_cache(
            make_request(100),
            block_size=16,
            num_local_computed_tokens=0,
            num_external_computed_tokens=0,
            load_kv_async=False,
        )
