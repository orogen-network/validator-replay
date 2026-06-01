"""Validator replay tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from mining_types import FaultCode, Receipt, SlashingEvidence, generate_keypair

from validator_replay import OperatorRegistry, ValidatorConfig, build_app
from validator_replay.chain import (
    ChainRpcClient,
    ChainUnreachableError,
    _decode_compact,
    _encode_compact,
    _encode_slashing_extrinsic,
    _to_32,
)
from validator_replay.replay import (
    MaterialMismatchTracker,
    ReplayMode,
    ReplayVerdict,
    TokenEvidence,
    logprob_agreement,
    replay_receipt,
    resolved_replay_mode,
    resolved_tolerance,
    token_overlap_ratio,
    tolerant_score,
)
from validator_replay.sampler import CommitReveal, StakeWeightedSampler


@pytest.fixture(autouse=True)
def _allow_plaintext_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests use http://example.test and ws://127.0.0.1:1 (loopback);
    plaintext is fine for the loopback case, and we explicitly opt-in for
    the non-loopback test fixtures."""
    monkeypatch.setenv("VALIDATOR_ALLOW_PLAINTEXT_RPC", "1")
    # N-W-01: allow unsigned receipts by default in test (most tests don't
    # care about sig-verification); targeted tests opt OUT of this via
    # monkeypatch.delenv when they want to exercise the verification path.
    monkeypatch.setenv("ALLOW_UNSIGNED_INGEST", "1")


def _make_receipt(op: str, job: str, response_hash: str = "rs", model_weight: str = "w") -> Receipt:
    return Receipt(
        job_id=job, operator_id=op, model_id="m",
        model_weight_hash=model_weight, customer_nonce="n",
        request_hash="rq", response_hash=response_hash,
        kernel_pack_hash="k", attestation_report_hash="a",
        timestamp_ms=1, gateway_id="gw",
    )


def _replay_input() -> dict[str, object]:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "seed": 0,
    }


@pytest.fixture
def config() -> ValidatorConfig:
    priv, _ = generate_keypair()
    return ValidatorConfig(
        validator_id="val-1", validator_private_key_hex=priv, sample_rate=0.5,
    )


def test_commit_reveal_determinism() -> None:
    c = CommitReveal(epoch_seed="seed-1", validator_id="v", salt="s")
    assert c.commit() == c.commit()


def test_sampler_respects_rate() -> None:
    s = StakeWeightedSampler(sample_rate=0.5)
    receipts = [_make_receipt(f"op-{i}", f"j-{i}") for i in range(10)]
    chosen = s.sample(receipts, CommitReveal("ep", "v", "salt"))
    assert 1 <= len(chosen) <= 10
    # determinism with same commit
    again = s.sample(receipts, CommitReveal("ep", "v", "salt"))
    assert [r.job_id for r in chosen] == [r.job_id for r in again]


def test_sampler_stake_weighting_biases_toward_heavy() -> None:
    s = StakeWeightedSampler(
        sample_rate=0.5,
        operator_stake={"heavy": 100, "light": 1},
    )
    receipts: list[Receipt] = []
    for i in range(20):
        op = "heavy" if i < 10 else "light"
        receipts.append(_make_receipt(op, f"j-{i}"))
    chosen = s.sample(receipts, CommitReveal("ep", "v", "salt"))
    heavy = sum(1 for r in chosen if r.operator_id == "heavy")
    # heavy should win ≥ half of the chosen — sanity check, not statistical guarantee.
    assert heavy >= len(chosen) // 2


def test_replay_match_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_REPLAY", "1")
    r = _make_receipt("op", "j-1")
    assert replay_receipt(r).verdict == ReplayVerdict.MATCH


def test_replay_without_worker_is_skipped_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VALIDATOR_ALLOW_STUB_REPLAY", raising=False)
    r = _make_receipt("op", "j-no-worker")
    out = replay_receipt(r)
    assert out.verdict == ReplayVerdict.SKIPPED
    assert "worker_replay_url required" in out.detail


def test_replay_response_mismatch() -> None:
    r = _make_receipt("op", "j-1", response_hash="rs")
    out = replay_receipt(r, expected_response_hash="WRONG")
    assert out.verdict == ReplayVerdict.MISMATCH
    assert out.fault and out.fault.value == "WrongResponse"


def test_replay_model_mismatch() -> None:
    r = _make_receipt("op", "j-1", model_weight="w-A")
    out = replay_receipt(r, expected_model_weight_hash="w-B")
    assert out.verdict == ReplayVerdict.MISMATCH
    assert out.fault and out.fault.value == "WrongModel"


# ------------------------------------------------------------------ tolerant mode


def test_replay_mode_defaults_to_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALIDATOR_REPLAY_MODE", raising=False)
    assert resolved_replay_mode() is ReplayMode.EXACT


def test_replay_mode_resolves_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALIDATOR_REPLAY_MODE", "tolerant")
    assert resolved_replay_mode() is ReplayMode.TOLERANT


def test_replay_mode_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown VALIDATOR_REPLAY_MODE"):
        resolved_replay_mode("loose")


def test_tolerance_defaults_and_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VALIDATOR_REPLAY_TOLERANCE", raising=False)
    assert resolved_tolerance() == 0.98
    assert resolved_tolerance(2.0) == 1.0
    assert resolved_tolerance(-1.0) == 0.0


def test_token_overlap_and_logprob_agreement() -> None:
    assert token_overlap_ratio([], []) == 1.0
    assert token_overlap_ratio([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert token_overlap_ratio([1, 2, 3, 4], [1, 2, 9, 9]) == 0.5
    # length mismatch penalised against the longer sequence
    assert token_overlap_ratio([1, 2], [1, 2, 3, 4]) == 0.5
    assert logprob_agreement([-0.1, -0.2], [-0.1, -0.25]) == 1.0
    assert logprob_agreement([-0.1, -0.2], [-5.0, -0.2]) == 0.5


def test_tolerant_near_identical_is_match() -> None:
    r = _make_receipt("op", "j-tol-1", response_hash="exact-would-fail")
    claimed = {
        "token_ids": list(range(100)),
        "top_logprobs": [-0.1] * 100,
    }
    # One token differs out of 100 -> 0.99 overlap, above the 0.98 default.
    recomputed = {
        "token_ids": [*range(99), 9999],
        "top_logprobs": [-0.1] * 100,
    }
    out = replay_receipt(
        r,
        mode=ReplayMode.TOLERANT,
        expected_response_hash="different-hash-on-purpose",
        claimed_token_evidence=claimed,
        recomputed_token_evidence=recomputed,
    )
    assert out.verdict == ReplayVerdict.MATCH
    assert out.score is not None and out.score >= 0.98


def test_tolerant_very_different_is_mismatch() -> None:
    r = _make_receipt("op", "j-tol-2", response_hash="rs")
    claimed = {"token_ids": list(range(100)), "top_logprobs": [-0.1] * 100}
    recomputed = {
        "token_ids": list(range(1000, 1100)),  # entirely different
        "top_logprobs": [-4.0] * 100,
    }
    out = replay_receipt(
        r,
        mode=ReplayMode.TOLERANT,
        expected_response_hash="rs",  # exact would MATCH; tolerant must MISMATCH
        claimed_token_evidence=claimed,
        recomputed_token_evidence=recomputed,
    )
    assert out.verdict == ReplayVerdict.MISMATCH
    assert out.fault and out.fault.value == "WrongResponse"
    assert out.score is not None and out.score < 0.98


def test_tolerant_falls_back_to_exact_without_evidence() -> None:
    """Tolerant mode with no token evidence must not be weaker than exact."""
    r = _make_receipt("op", "j-tol-3", response_hash="rs")
    out = replay_receipt(r, mode=ReplayMode.TOLERANT, expected_response_hash="WRONG")
    assert out.verdict == ReplayVerdict.MISMATCH
    assert out.fault and out.fault.value == "WrongResponse"


def test_tolerant_score_takes_worse_signal() -> None:
    claimed = TokenEvidence(token_ids=[1, 2, 3, 4], top_logprobs=[-0.1, -0.1, -0.1, -0.1])
    # Tokens fully agree but logprobs all diverge.
    recomputed = TokenEvidence(token_ids=[1, 2, 3, 4], top_logprobs=[-9.0, -9.0, -9.0, -9.0])
    assert tolerant_score(claimed, recomputed) == 0.0


def test_material_mismatch_tracker() -> None:
    t = MaterialMismatchTracker(threshold=3)
    assert not t.is_material("op-x")
    t.record("op-x")
    t.record("op-x")
    assert not t.is_material("op-x")
    t.record("op-x")
    assert t.is_material("op-x")
    t.reset("op-x")
    assert not t.is_material("op-x")


@respx.mock
def test_tolerant_worker_replay_uses_full_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: in tolerant mode the validator pulls the worker's token
    evidence (full body) as the recomputed side."""
    monkeypatch.setenv("VALIDATOR_REPLAY_MODE", "tolerant")
    receipt = _make_receipt("op-1", "j-tol-worker", response_hash="claimed-hash")
    respx.post("http://worker.test/v1/replay").mock(
        return_value=httpx.Response(
            200,
            json={
                "response_hash": "fresh-hash",  # differs, but tolerant ignores it
                "token_ids": [1, 2, 3, 4],
                "top_logprobs": [-0.1, -0.1, -0.1, -0.1],
            },
        )
    )
    out = replay_receipt(
        receipt,
        worker_url="http://worker.test",
        replay_input={
            **_replay_input(),
            "claimed_token_evidence": {
                "token_ids": [1, 2, 3, 4],
                "top_logprobs": [-0.1, -0.1, -0.1, -0.1],
            },
        },
    )
    assert out.verdict == ReplayVerdict.MATCH
    assert out.score == 1.0


def test_healthz(config: ValidatorConfig) -> None:
    app = build_app(config)
    with TestClient(app) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["replay_input_policy"] == "ephemeral"
        assert body["replay_input_max_bytes"] > 0


def test_production_worker_replay_requires_explicit_replay_input_policy(
    config: ValidatorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OROGEN_ENV", "production")
    monkeypatch.delenv("VALIDATOR_REPLAY_INPUT_POLICY", raising=False)
    cfg = ValidatorConfig(
        validator_id=config.validator_id,
        validator_private_key_hex=config.validator_private_key_hex,
        worker_replay_url="http://worker.test",
    )
    with pytest.raises(RuntimeError, match="VALIDATOR_REPLAY_INPUT_POLICY=ephemeral"):
        build_app(cfg)


def test_production_worker_replay_accepts_ephemeral_policy(
    config: ValidatorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OROGEN_ENV", "production")
    monkeypatch.setenv("VALIDATOR_REPLAY_INPUT_POLICY", "ephemeral")
    monkeypatch.setenv("VALIDATOR_API_TOKEN", "validator-secret")
    cfg = ValidatorConfig(
        validator_id=config.validator_id,
        validator_private_key_hex=config.validator_private_key_hex,
        worker_replay_url="http://worker.test",
    )
    app = build_app(cfg)
    with TestClient(app) as c:
        r = c.get(
            "/healthz",
            headers={"Authorization": "Bearer validator-secret"},
        )
        assert r.status_code == 200
        assert r.json()["replay_input_policy"] == "ephemeral"


def _hex_op(i: int) -> str:
    """Generate a 64-hex-char operator ID (H-08: chain enforces this shape)."""
    return f"{i:064x}"


def test_run_epoch_clean_pass(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    app = build_app(config)
    receipts = [_make_receipt(_hex_op(i), f"j-{i}") for i in range(8)]
    with TestClient(app) as c:
        r = c.post(
            "/run_epoch",
            json={
                "receipts": [x.model_dump(mode="json") for x in receipts],
                "epoch_seed": "seed-1",
                "operator_stake": {_hex_op(i): 10 for i in range(8)},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["slashings_submitted"] == 0
        # C-04: per-epoch shape distinguishes chain vs local queue.
        assert "submitted_to_chain" in body
        assert "queued_locally" in body


def test_run_epoch_queues_slashing_on_worker_replay_mismatch(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    monkeypatch.setattr(
        "validator_replay.replay.replay_via_worker",
        lambda _receipt, _worker_url, _replay_input: "honest",
    )
    cfg = ValidatorConfig(
        validator_id=config.validator_id,
        validator_private_key_hex=config.validator_private_key_hex,
        sample_rate=1.0,
        worker_replay_url="http://worker.test",
        replay_input_policy="ephemeral",
    )
    app = build_app(cfg)
    bad = "ff" * 32
    receipt = _make_receipt(bad, "j-1", response_hash="dishonest")
    with TestClient(app) as c:
        r = c.post(
            "/run_epoch",
            json={
                "receipts": [receipt.model_dump(mode="json")],
                "epoch_seed": "seed-1",
                "operator_stake": {bad: 1000},
                "replay_inputs": {"j-1": _replay_input()},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["sampled_count"] == 1
        assert body["slashings_submitted"] == 1
        assert body["queued_locally"] == 1
        assert body["verdicts"][0]["verdict"] == "mismatch"
        assert body["verdicts"][0]["fault"] == "WrongResponse"


def test_run_epoch_returns_503_when_chain_unreachable_and_no_stake_provided(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-04: /run_epoch without explicit operator_stake should return 503
    if chain is unreachable and stub fallback is NOT opted into."""
    monkeypatch.delenv("VALIDATOR_ALLOW_STUB_CHAIN", raising=False)
    cfg = ValidatorConfig(
        validator_id=config.validator_id,
        validator_private_key_hex=config.validator_private_key_hex,
        chain_rpc_url="ws://127.0.0.1:1",
        sample_rate=config.sample_rate,
    )
    app = build_app(cfg)
    receipts = [_make_receipt(_hex_op(i), f"j-{i}") for i in range(3)]
    with TestClient(app) as c:
        r = c.post(
            "/run_epoch",
            json={
                "receipts": [x.model_dump(mode="json") for x in receipts],
                "epoch_seed": "seed-503",
                # no operator_stake — triggers chain RPC.
            },
        )
        assert r.status_code == 503


# ------------------------------------------------------------------ chain RPC


def test_chain_rpc_raises_on_unreachable_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-04: chain RPC failure must RAISE, not silently substitute stub data."""
    monkeypatch.delenv("VALIDATOR_ALLOW_STUB_CHAIN", raising=False)
    client = ChainRpcClient("ws://127.0.0.1:1")  # closed
    with pytest.raises(ChainUnreachableError):
        client.get_operator_stakes()


def test_chain_rpc_falls_back_to_stub_when_opted_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """C-04: stub fallback is opt-in via VALIDATOR_ALLOW_STUB_CHAIN=1."""
    stub_dir = tmp_path / "stub_data"
    stub_dir.mkdir()
    (stub_dir / "operators.json").write_text(
        '[{"operator_id": "' + "ab" * 32 + '", "stake": 1000, "active": true}]'
    )
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    monkeypatch.setenv("VALIDATOR_STUB_DATA_DIR", str(stub_dir))
    client = ChainRpcClient("ws://127.0.0.1:1")
    ops = client.get_operator_stakes()
    assert len(ops) == 1
    assert ops[0].stake == 1000


def test_chain_rpc_no_autowrite_stub_on_import() -> None:
    """L-05: importing validator_replay.chain must NOT write stub data
    to the package install path."""
    import validator_replay.chain as chain_mod

    stub_dir = Path(chain_mod.__file__).resolve().parent / "_stub_data"
    if stub_dir.exists():
        # If a tmp test wrote here, that's a bug; we want NO file ever.
        assert not (stub_dir / "operators.json").exists()


def test_to_32_rejects_non_hex_input() -> None:
    """H-08: _to_32 must refuse to zero-pad non-hex operator IDs."""
    with pytest.raises(ValueError):
        _to_32("op-bad")
    with pytest.raises(ValueError):
        _to_32("stub-op-1")


def test_to_32_accepts_64_char_hex() -> None:
    out = _to_32("ab" * 32)
    assert out == bytes.fromhex("ab" * 32)
    out2 = _to_32("0x" + "cd" * 32)
    assert out2 == bytes.fromhex("cd" * 32)


def test_chain_rpc_uses_real_response_when_reachable() -> None:
    """When the RPC endpoint responds, decode the operators payload."""
    with respx.mock(base_url="http://example.test") as router:
        # Probe succeeds.
        router.post(
            "/", name="all"
        ).mock(
            side_effect=[
                httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0x"}),
                httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": _encode_operators_hex(
                            [(b"\x11" * 32, 9_999), (b"\x22" * 32, 42_000)]
                        ),
                    },
                ),
            ]
        )
        client = ChainRpcClient("http://example.test/")
        ops = client.get_operator_stakes()
    assert {(o.operator_id, o.stake) for o in ops} == {
        ("11" * 32, 9_999),
        ("22" * 32, 42_000),
    }


def test_chain_rpc_submits_slashing_extrinsic_when_reachable() -> None:
    ev = SlashingEvidence(
        operator_id="11" * 32,
        fault_code=FaultCode.WRONG_RESPONSE,
        evidence_hash="22" * 32,
        related_job_id="33" * 32,
        related_receipt_hash="44" * 32,
    )
    with respx.mock(base_url="http://example.test") as router:
        route = router.post(
            "/", name="all"
        ).mock(
            side_effect=[
                httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0x"}),
                httpx.Response(
                    200, json={"jsonrpc": "2.0", "id": 1, "result": "0xdeadbeef"}
                ),
            ]
        )
        client = ChainRpcClient("http://example.test/")
        tx_hash = client.submit_slashing_evidence(ev)
    assert tx_hash == "0xdeadbeef"
    # Second call carried the SCALE-encoded extrinsic.
    submit_body = route.calls[1].request.content.decode()
    expected_hex = _encode_slashing_extrinsic(ev).hex()
    assert expected_hex in submit_body


def test_slashing_extrinsic_encoding_matches_rfc0005_shape() -> None:
    ev = SlashingEvidence(
        operator_id="0x" + "55" * 32,
        fault_code=FaultCode.QUANTIZATION_SWAP,
        evidence_hash="0x" + "ee" * 32,
        related_job_id=None,
        related_receipt_hash="0x" + "ff" * 32,
    )
    body = _encode_slashing_extrinsic(ev)
    # pallet/call index + 32-byte op + 1-byte fault + 32-byte evidence
    # + 1-byte Option(None) tag + (1+32) Option(Some) = 2 + 32 + 1 + 32 + 1 + 33
    assert len(body) == 101
    assert body[0] == 42  # PALLET_INDEX
    assert body[1] == 0  # CALL_INDEX
    # Fault-code index: WrongModel=0, WrongResponse=1, ..., QuantizationSwap=4.
    assert body[2 + 32] == 4
    # Option<None> tag for related_job_id at offset 2+32+1+32 = 67
    assert body[67] == 0
    # Option<Some> tag for related_receipt_hash at offset 68
    assert body[68] == 1


def test_scale_compact_roundtrip() -> None:
    for v in (0, 1, 63, 64, 16_383, 16_384, 100_000):
        encoded = _encode_compact(v)
        decoded, offset = _decode_compact(encoded, 0)
        assert decoded == v
        assert offset == len(encoded)


def _encode_operators_hex(operators: list[tuple[bytes, int]]) -> str:
    """Helper: encode a `Vec<(AccountId, Balance)>` as the chain would."""
    out = bytearray(_encode_compact(len(operators)))
    for account, balance in operators:
        assert len(account) == 32
        out.extend(account)
        out.extend(int.to_bytes(balance, 16, "little"))
    return "0x" + out.hex()


# ------------------------------------------------------------------ worker replay

@pytest.fixture
def worker_config() -> ValidatorConfig:
    priv, _ = generate_keypair()
    return ValidatorConfig(
        validator_id="val-1",
        validator_private_key_hex=priv,
        sample_rate=1.0,  # so we deterministically replay every receipt
        worker_replay_url="http://worker.test",
    )


@respx.mock
def test_worker_replay_path_matches_when_worker_agrees(
    worker_config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATOR_WORKER_API_TOKEN", "worker-secret")
    receipt = _make_receipt("op-1", "j-w-match", response_hash="abc123")

    def _worker(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer worker-secret"
        return httpx.Response(200, json={"response_hash": "abc123"})

    respx.post("http://worker.test/v1/replay").mock(side_effect=_worker)
    app = build_app(worker_config)
    with TestClient(app) as c:
        r = c.post(
            "/replay",
            json={
                "receipts": [receipt.model_dump(mode="json")],
                "replay_inputs": {receipt.job_id: _replay_input()},
            },
        )
        assert r.status_code == 200
        out = r.json()["results"][0]
        assert out["verdict"] == "match"


@respx.mock
def test_worker_replay_path_mismatches_when_worker_disagrees(
    worker_config: ValidatorConfig,
) -> None:
    receipt = _make_receipt("op-1", "j-w-miss", response_hash="claimed-X")
    respx.post("http://worker.test/v1/replay").mock(
        return_value=httpx.Response(200, json={"response_hash": "actually-Y"})
    )
    app = build_app(worker_config)
    with TestClient(app) as c:
        r = c.post(
            "/replay",
            json={
                "receipts": [receipt.model_dump(mode="json")],
                "replay_inputs": {receipt.job_id: _replay_input()},
            },
        )
        assert r.status_code == 200
        out = r.json()["results"][0]
        assert out["verdict"] == "mismatch"
        assert out["fault"] == "WrongResponse"


@respx.mock
def test_worker_replay_path_mismatches_when_worker_rejects_replay_input(
    worker_config: ValidatorConfig,
) -> None:
    receipt = _make_receipt("op-1", "j-w-reject", response_hash="claimed-X")
    respx.post("http://worker.test/v1/replay").mock(
        return_value=httpx.Response(400, json={"detail": "request_hash mismatch"})
    )
    app = build_app(worker_config)
    with TestClient(app) as c:
        r = c.post(
            "/replay",
            json={
                "receipts": [receipt.model_dump(mode="json")],
                "replay_inputs": {receipt.job_id: _replay_input()},
            },
        )
        assert r.status_code == 200
        out = r.json()["results"][0]
        assert out["verdict"] == "mismatch"
        assert out["fault"] == "WrongResponse"


@respx.mock
def test_worker_replay_path_skips_when_worker_unreachable(
    worker_config: ValidatorConfig,
) -> None:
    receipt = _make_receipt("op-1", "j-w-down", response_hash="x")
    respx.post("http://worker.test/v1/replay").mock(
        side_effect=httpx.ConnectError("nope")
    )
    res = replay_receipt(
        receipt,
        worker_url=worker_config.worker_replay_url,
        replay_input=_replay_input(),
    )
    assert res.verdict == ReplayVerdict.SKIPPED


def test_worker_replay_requires_original_input(worker_config: ValidatorConfig) -> None:
    receipt = _make_receipt("op-1", "j-no-input", response_hash="x")
    res = replay_receipt(receipt, worker_url=worker_config.worker_replay_url)
    assert res.verdict == ReplayVerdict.SKIPPED
    assert "replay input required" in res.detail


def test_run_epoch_queues_slashing_when_worker_rejects_replay_input(
    worker_config: ValidatorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    request = httpx.Request("POST", "http://worker.test/v1/replay")
    response = httpx.Response(
        400,
        json={"detail": "request_hash mismatch"},
        request=request,
    )
    monkeypatch.setattr(
        "validator_replay.replay.replay_via_worker",
        lambda _receipt, _worker_url, _replay_input: (_ for _ in ()).throw(
            httpx.HTTPStatusError("request_hash mismatch", request=request, response=response)
        ),
    )
    receipt = _make_receipt("op-1", "j-w-reject-epoch", response_hash="claimed-X")
    app = build_app(worker_config)
    with TestClient(app) as c:
        r = c.post(
            "/run_epoch",
            json={
                "epoch_seed": "seed",
                "receipts": [receipt.model_dump(mode="json")],
                "operator_stake": {"op-1": 100},
                "replay_inputs": {receipt.job_id: _replay_input()},
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["verdicts"][0]["verdict"] == "mismatch"
        assert body["verdicts"][0]["fault"] == "WrongResponse"
        assert body["queued_locally"] == 1


@respx.mock
def test_worker_replay_input_cannot_override_receipt_fields(
    worker_config: ValidatorConfig,
) -> None:
    receipt = _make_receipt("op-1", "j-bound", response_hash="ok")

    def _worker(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["job_id"] == "j-bound"
        assert payload["model_id"] == "m"
        assert payload["customer_nonce"] == "n"
        assert payload["request_hash"] == "rq"
        return httpx.Response(200, json={"response_hash": "ok"})

    respx.post("http://worker.test/v1/replay").mock(side_effect=_worker)
    res = replay_receipt(
        receipt,
        worker_url=worker_config.worker_replay_url,
        replay_input={
            **_replay_input(),
            "job_id": "attacker-job",
            "model_id": "attacker-model",
            "customer_nonce": "attacker-nonce",
            "request_hash": "attacker-hash",
        },
    )
    assert res.verdict == ReplayVerdict.MATCH


def test_replay_inputs_are_size_bounded_by_ephemeral_policy(
    worker_config: ValidatorConfig,
) -> None:
    cfg = ValidatorConfig(
        validator_id=worker_config.validator_id,
        validator_private_key_hex=worker_config.validator_private_key_hex,
        sample_rate=worker_config.sample_rate,
        worker_replay_url=worker_config.worker_replay_url,
        replay_input_policy="ephemeral",
        replay_input_max_bytes=32,
    )
    receipt = _make_receipt("op-1", "j-too-big", response_hash="x")
    app = build_app(cfg)
    with TestClient(app) as c:
        r = c.post(
            "/replay",
            json={
                "receipts": [receipt.model_dump(mode="json")],
                "replay_inputs": {
                    receipt.job_id: {
                        "messages": [{"role": "user", "content": "x" * 128}],
                        "max_tokens": 8,
                        "seed": 0,
                    }
                },
            },
        )
        assert r.status_code == 413


@respx.mock
def test_replay_inputs_are_not_returned_by_api(worker_config: ValidatorConfig) -> None:
    receipt = _make_receipt("op-1", "j-private", response_hash="ok")
    respx.post("http://worker.test/v1/replay").mock(
        return_value=httpx.Response(200, json={"response_hash": "ok"})
    )
    app = build_app(worker_config)
    with TestClient(app) as c:
        r = c.post(
            "/replay",
            json={
                "receipts": [receipt.model_dump(mode="json")],
                "replay_inputs": {
                    receipt.job_id: {
                        "messages": [{"role": "user", "content": "private prompt"}],
                        "max_tokens": 8,
                        "seed": 0,
                    }
                },
            },
        )
        assert r.status_code == 200
        assert "private prompt" not in r.text
        assert "replay_inputs" not in r.text


# ------------------------------------------------------------------ /internal/operators

def test_operators_endpoint_returns_503_when_chain_unreachable(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C-04: chain RPC failure must surface as a 503, not as a silent stub."""
    monkeypatch.delenv("VALIDATOR_ALLOW_STUB_CHAIN", raising=False)
    cfg = ValidatorConfig(
        validator_id=config.validator_id,
        validator_private_key_hex=config.validator_private_key_hex,
        chain_rpc_url="ws://127.0.0.1:1",  # closed
    )
    app = build_app(cfg)
    with TestClient(app) as c:
        r = c.get("/internal/operators")
        assert r.status_code == 503


def test_operators_endpoint_returns_stub_when_opted_in(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    stub_dir = tmp_path / "stub_data"
    stub_dir.mkdir()
    (stub_dir / "operators.json").write_text(
        '[{"operator_id": "' + "ab" * 32 + '", "stake": 1000, "active": true}]'
    )
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    monkeypatch.setenv("VALIDATOR_STUB_DATA_DIR", str(stub_dir))
    cfg = ValidatorConfig(
        validator_id=config.validator_id,
        validator_private_key_hex=config.validator_private_key_hex,
        chain_rpc_url="ws://127.0.0.1:1",
    )
    app = build_app(cfg)
    with TestClient(app) as c:
        r = c.get("/internal/operators")
        assert r.status_code == 200
        body = r.json()
        assert body["rpc_url"] == "ws://127.0.0.1:1"
        assert len(body["operators"]) == 1


def test_healthz_surfaces_chain_rpc_url(config: ValidatorConfig) -> None:
    app = build_app(config)
    with TestClient(app) as c:
        r = c.get("/healthz")
        body = r.json()
        assert body["ok"] is True
        assert "chain_rpc_url" in body


# ------------------------------------------------------------------ N-W-01

def test_run_epoch_rejects_unsigned_receipts_in_strict_mode(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N-W-01: with strict-sig mode on, unknown operators are filtered out
    and surfaced in the response counters."""
    monkeypatch.delenv("ALLOW_UNSIGNED_INGEST", raising=False)
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    reg = OperatorRegistry()  # empty registry — nothing trusted
    app = build_app(config, registry=reg)
    receipts = [_make_receipt(_hex_op(i), f"j-{i}") for i in range(3)]
    with TestClient(app) as c:
        r = c.post(
            "/run_epoch",
            json={
                "receipts": [x.model_dump(mode="json") for x in receipts],
                "epoch_seed": "seed-strict",
                "operator_stake": {_hex_op(i): 10 for i in range(3)},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rejected_unknown_operator"] == 3
        assert body["rejected_invalid_signature"] == 0
        # Nothing got past the gate — no verdicts.
        assert body["verdicts"] == []


def test_run_epoch_accepts_properly_signed_receipts(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N-W-01: receipts signed by the registered operator key pass the
    verification gate and are sampled/replayed normally."""
    from mining_types import generate_keypair

    monkeypatch.delenv("ALLOW_UNSIGNED_INGEST", raising=False)
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    op_priv, op_pub = generate_keypair()
    op_id = _hex_op(7)
    reg = OperatorRegistry()
    reg.register(op_id, op_pub)
    app = build_app(config, registry=reg)
    receipts = [_make_receipt(op_id, f"j-{i}").sign(op_priv) for i in range(3)]
    with TestClient(app) as c:
        r = c.post(
            "/run_epoch",
            json={
                "receipts": [x.model_dump(mode="json") for x in receipts],
                "epoch_seed": "seed-strict-ok",
                "operator_stake": {op_id: 10},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rejected_unknown_operator"] == 0
        assert body["rejected_invalid_signature"] == 0


def test_run_epoch_rejects_tampered_signature(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N-W-01: a registered operator + tampered signature → invalid-sig count."""
    from mining_types import generate_keypair

    monkeypatch.delenv("ALLOW_UNSIGNED_INGEST", raising=False)
    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    op_priv, op_pub = generate_keypair()
    op_id = _hex_op(8)
    reg = OperatorRegistry()
    reg.register(op_id, op_pub)
    app = build_app(config, registry=reg)
    good = _make_receipt(op_id, "j-good").sign(op_priv)
    bad = good.model_copy(update={"operator_signature": "00" * 64})
    with TestClient(app) as c:
        r = c.post(
            "/run_epoch",
            json={
                "receipts": [good.model_dump(mode="json"), bad.model_dump(mode="json")],
                "epoch_seed": "seed-strict-tamper",
                "operator_stake": {op_id: 10},
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rejected_invalid_signature"] == 1


def test_sample_surfaces_rejection_counts(
    config: ValidatorConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N-W-01: /sample also drops bad receipts and reports the counts."""
    monkeypatch.delenv("ALLOW_UNSIGNED_INGEST", raising=False)
    reg = OperatorRegistry()  # empty
    app = build_app(config, registry=reg)
    receipts = [_make_receipt(_hex_op(i), f"j-{i}") for i in range(5)]
    with TestClient(app) as c:
        r = c.post(
            "/sample",
            json={
                "receipts": [x.model_dump(mode="json") for x in receipts],
                "epoch_seed": "ep-sample",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["rejected_unknown_operator"] == 5
        assert body["sampled"] == []
