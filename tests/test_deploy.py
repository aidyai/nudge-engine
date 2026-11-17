"""The Modal files run in the cloud, but their logic is checked here so mistakes show up before a deploy."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from deploy import llm, tts


def test_vllm_command_follows_the_model_card_recipe():
    cmd = llm.build_vllm_command("a-key")
    assert cmd[:3] == ["vllm", "serve", "pipecat-ai/phonellm-mini-alpha-2"]
    pairs = dict(zip(cmd, cmd[1:], strict=False))
    assert pairs["--dtype"] == "bfloat16"
    assert pairs["--max-model-len"] == "32768"
    assert pairs["--gpu-memory-utilization"] == "0.90"
    assert pairs["--max-num-seqs"] == "256"
    assert pairs["--mamba-ssm-cache-dtype"] == "float32"
    assert pairs["--tool-call-parser"] == "qwen3_coder"
    assert pairs["--served-model-name"] == "phonellm-mini-alpha-2"
    for flag in (
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--async-scheduling",
        "--enable-auto-tool-choice",
    ):
        assert flag in cmd


def test_vllm_command_avoids_the_two_flags_the_card_warns_about():
    cmd = llm.build_vllm_command("a-key")
    assert "--enforce-eager" not in cmd  # CUDA graphs are 5 to 7 times faster decoding
    assert "--reasoning-parser" not in cmd  # thinking is off, a parser would hide tool calls


def test_vllm_is_always_started_with_the_api_key():
    cmd = llm.build_vllm_command("a-key")
    assert cmd[cmd.index("--api-key") + 1] == "a-key"


def test_revision_is_pinned_only_when_given():
    assert "--revision" not in llm.build_vllm_command("k")
    cmd = llm.build_vllm_command("k", revision="abc123")
    assert cmd[cmd.index("--revision") + 1] == "abc123"


@pytest.mark.parametrize(
    ("header", "key", "ok"),
    [
        ("Bearer s3cret", "s3cret", True),
        ("Bearer wrong", "s3cret", False),
        ("s3cret", "s3cret", False),  # missing the word Bearer
        (None, "s3cret", False),
        ("Bearer ", "s3cret", False),
        ("Bearer anything", "", False),  # an unset server key must never let everyone in
    ],
)
def test_speech_endpoint_authorization(header, key, ok):
    assert tts.is_authorized(header, key) is ok


@given(st.lists(st.floats(min_value=-3, max_value=3, allow_nan=False), min_size=0, max_size=500))
def test_pcm_conversion_is_two_bytes_per_sample_and_never_overflows(samples):
    data = tts.float_to_pcm16(np.array(samples, dtype=np.float32))
    assert len(data) == 2 * len(samples)
    decoded = np.frombuffer(data, dtype="<i2")
    assert decoded.size == 0 or (decoded.min() >= -32767 and decoded.max() <= 32767)


def test_pcm_conversion_known_values():
    data = tts.float_to_pcm16(np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32))
    assert list(np.frombuffer(data, dtype="<i2")) == [0, 32767, -32767, 16383]


def test_speak_request_rejects_empty_and_oversized_text():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        tts.SpeakRequest(text="")
    with pytest.raises(ValidationError):
        tts.SpeakRequest(text="x" * (tts.MAX_CHARS + 1))
    assert tts.SpeakRequest(text="Hello.").text == "Hello."


def test_logs_refuse_text_and_secrets():
    with pytest.raises(ValueError):
        tts.log_event("tts_request", text="a borrower's sentence")
    with pytest.raises(ValueError):
        tts.log_event("tts_request", authorization="Bearer abc")


def test_a_log_line_is_one_line_of_json(caplog):
    import json
    import logging

    with caplog.at_level(logging.INFO, logger="nudge.tts"):
        tts.log_event("tts_request", chars=12, latency_ms=340)
    record = [r for r in caplog.records if r.name == "nudge.tts"][-1]
    assert json.loads(record.getMessage()) == {"event": "tts_request", "chars": 12, "latency_ms": 340}


def test_idle_cost_rules_are_stated_in_the_files():
    # The numbers live in the decorators; this guards against someone quietly turning a cap off.
    for path in ("deploy/llm.py", "deploy/tts.py"):
        source = open(path).read()
        assert "max_containers=1" in source
        assert "scaledown_window=2 * MINUTES" in source
    assert llm.SERVER["min_containers"] == 0  # nothing runs while idle, unless LLM_MIN_CONTAINERS is set


def test_voice_weights_are_cached_and_the_model_is_warmed_up():
    import inspect

    src = inspect.getsource(tts)
    assert 'HF_HOME": "/cache/huggingface"' in src and "chatterbox-cache" in src
    assert "weights.commit()" in src and 'self.synthesize("Warming up.")' in src


def test_sleep_mode_is_only_added_when_asked():
    assert "--enable-sleep-mode" not in llm.build_vllm_command("k")
    assert "--enable-sleep-mode" in llm.build_vllm_command("k", sleep_mode=True)


def test_triton_kernels_stay_out_of_the_volume():
    """Measured: putting TRITON_CACHE_DIR in the volume doubled the cold start (9 to 18 minutes) and broke torch's cache."""
    assert "TRITON_CACHE_DIR" not in llm.IMAGE_ENV


def test_startup_may_take_longer_than_the_slowest_cold_start_we_saw():
    assert int(llm.SERVER["startup_timeout"]) >= 45 * 60  # pyright: ignore[reportArgumentType]  # starts took 9, 18 and 29 minutes


def test_the_default_deploy_does_not_open_vllms_dev_endpoints(monkeypatch):
    """VLLM_SERVER_DEV_MODE adds /sleep and /wake_up, which the API key does not protect. Only the snapshot build has it."""
    import importlib

    assert "VLLM_SERVER_DEV_MODE" not in llm.IMAGE_ENV and llm.SNAPSHOT is False
    assert llm.SERVER["unauthenticated"] is True  # public URL, vLLM checks the API key on /v1
    monkeypatch.setenv("LLM_SNAPSHOT", "1")
    try:
        snap = importlib.reload(llm)
        assert snap.SNAPSHOT and snap.IMAGE_ENV["VLLM_SERVER_DEV_MODE"] == "1"
        assert snap.IMAGE_ENV["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        assert snap.SERVER["unauthenticated"] is False  # the dev endpoints have no key check, so Modal's proxy must guard the door
    finally:
        monkeypatch.delenv("LLM_SNAPSHOT")
        importlib.reload(llm)
    assert llm.SNAPSHOT is False


def test_one_replica_can_be_kept_warm_for_a_calling_window(monkeypatch):
    import importlib

    monkeypatch.setenv("LLM_MIN_CONTAINERS", "1")
    try:
        assert importlib.reload(llm).MIN_CONTAINERS == 1
    finally:
        monkeypatch.delenv("LLM_MIN_CONTAINERS")
        importlib.reload(llm)
    assert llm.MIN_CONTAINERS == 0 and llm.SERVER["max_containers"] == 1  # never more than one GPU


# ---- copying the weights to the container's own disk


def make_cache(tmp_path, files=None, revision="abc123"):
    repo = tmp_path / "hub" / "models--pipecat-ai--phonellm-mini-alpha-2"
    snap = repo / "snapshots" / revision
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(revision)
    blobs = repo / "blobs"
    blobs.mkdir()
    for name, data in (files or {"config.json": b"{}", "model-00001.safetensors": b"x" * 5000, "sub/tokenizer.json": b"t"}).items():
        (blobs / name.replace("/", "_")).write_bytes(data)
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).symlink_to(blobs / name.replace("/", "_"))  # the real cache links to blobs, like this
    return tmp_path / "hub"


def test_the_weights_are_copied_as_real_files_with_the_same_layout(tmp_path):
    root = make_cache(tmp_path)
    said = []
    out = llm.copy_weights(root, tmp_path / "local", say=said.append)
    assert out is not None and out == tmp_path / "local"
    assert (out / "model-00001.safetensors").read_bytes() == b"x" * 5000
    assert (out / "sub" / "tokenizer.json").read_bytes() == b"t"
    assert not (out / "config.json").is_symlink()
    assert any("copied in" in line for line in said)


def test_a_pinned_revision_picks_its_own_folder(tmp_path):
    root = make_cache(tmp_path, revision="old1")
    found = llm.find_snapshot(root, revision="old1")
    assert found is not None and found.name == "old1"


def test_with_nothing_downloaded_yet_it_falls_back_to_the_normal_start(tmp_path):
    said = []
    assert llm.copy_weights(tmp_path, tmp_path / "local", say=said.append) is None
    assert any("download" in line for line in said)


def test_it_does_not_copy_when_the_local_disk_is_too_small(tmp_path, monkeypatch):
    root = make_cache(tmp_path)
    monkeypatch.setattr(llm.shutil, "disk_usage", lambda _: type("D", (), {"free": 10})())
    assert llm.copy_weights(root, tmp_path / "local", say=lambda _: None) is None


def test_a_local_folder_is_served_without_a_revision():
    cmd = llm.build_vllm_command("k", revision="abc", model="/tmp/phonellm")
    assert cmd[2] == "/tmp/phonellm" and "--revision" not in cmd
    assert "--served-model-name" in cmd and llm.SERVED_NAME in cmd  # clients still ask for the same name
    assert "--revision" in llm.build_vllm_command("k", revision="abc")
