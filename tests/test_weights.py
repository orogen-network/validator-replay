"""Tests for Yuma weight-vector computation + encoding (reward-crank #3)."""

from __future__ import annotations

import pytest

from validator_replay.chain import OperatorStake
from validator_replay.weights import (
    U16_MAX,
    WeightVector,
    compute_weight_vector,
    encode_submit_weights_call,
    faulty_operators_from_verdicts,
)


def _op(account_hex: str, stake: int, active: bool = True) -> OperatorStake:
    return OperatorStake(operator_id=account_hex, stake=stake, active=active)


def test_compute_weight_vector_equal_weight_excludes_faulty() -> None:
    """Bootstrap default: clean operators get u16::MAX, faulty get 0 and are
    dropped from the vector entirely (a 0-weight entry would just pad it)."""
    ops = [
        _op("11" * 32, 9_999),
        _op("22" * 32, 42_000),
        _op("33" * 32, 1_000),
    ]
    vec = compute_weight_vector(
        epoch=7, operators=ops, faulty_operators={"22" * 32}, equal_weight=True,
    )
    assert vec.epoch == 7
    accounts = {a.hex() for a, _ in vec.entries}
    assert accounts == {"11" * 32, "33" * 32}
    assert all(w == U16_MAX for _, w in vec.entries)


def test_compute_weight_vector_stake_proportional_when_not_equal() -> None:
    ops = [
        _op("11" * 32, 100),
        _op("22" * 32, 50),
    ]
    vec = compute_weight_vector(
        epoch=1, operators=ops, faulty_operators=set(), equal_weight=False,
    )
    weights = dict(vec.entries)
    assert weights[bytes.fromhex("11" * 32)] == U16_MAX  # max stake -> max weight
    assert weights[bytes.fromhex("22" * 32)] == U16_MAX // 2  # half stake


def test_compute_weight_vector_excludes_inactive_operators() -> None:
    ops = [
        _op("11" * 32, 100, active=True),
        _op("22" * 32, 100, active=False),  # frozen -> excluded
    ]
    vec = compute_weight_vector(
        epoch=1, operators=ops, faulty_operators=set(),
    )
    assert {a.hex() for a, _ in vec.entries} == {"11" * 32}


def test_compute_weight_vector_empty_when_all_faulty() -> None:
    ops = [_op("11" * 32, 100), _op("22" * 32, 100)]
    vec = compute_weight_vector(
        epoch=1, operators=ops,
        faulty_operators={"11" * 32, "22" * 32},
    )
    assert vec.entries == []


def test_compute_weight_vector_empty_when_no_operators() -> None:
    vec = compute_weight_vector(epoch=1, operators=[], faulty_operators=set())
    assert vec.entries == []


def test_encode_submit_weights_call_matches_runtime_abi() -> None:
    """Body = pallet(13) || call(0) || u64 epoch LE || compact len || (32B || u16 LE)*."""
    vec = WeightVector(
        epoch=42,
        entries=[(bytes.fromhex("11" * 32), 100), (bytes.fromhex("22" * 32), U16_MAX)],
    )
    body = encode_submit_weights_call(42, vec)
    assert body[0] == 13  # YumaConsensus pallet index
    assert body[1] == 0   # submit_weights call index
    # epoch u64 little-endian at offset 2
    assert int.from_bytes(body[2:10], "little") == 42
    # compact vec length at offset 10: 2 << 2 = 8
    assert body[10] == (2 << 2)
    # first account bytes start at offset 11
    assert body[11:43].hex() == "11" * 32
    assert int.from_bytes(body[43:45], "little") == 100
    # second account at 45
    assert body[45:77].hex() == "22" * 32
    assert int.from_bytes(body[77:79], "little") == U16_MAX
    # total length: 2 + 8 + 1 + (32+2)*2 = 77... recalc: 2+8+1+34*2 = 77
    assert len(body) == 2 + 8 + 1 + 34 * 2


def test_encode_submit_weights_call_rejects_non_32_byte_account() -> None:
    vec = WeightVector(epoch=0, entries=[(b"\x11" * 31, 1)])
    with pytest.raises(ValueError, match="32 bytes"):
        encode_submit_weights_call(0, vec)


def test_encode_submit_weights_call_clamps_weights_to_u16() -> None:
    vec = WeightVector(epoch=0, entries=[(bytes.fromhex("11" * 32), 999_999)])
    body = encode_submit_weights_call(0, vec)
    # 999_999 clamped to U16_MAX
    assert int.from_bytes(body[11 + 32 : 11 + 34], "little") == U16_MAX


def test_faulty_operators_from_verdicts_collects_mismatches() -> None:
    verdicts = [
        {"job_id": "j1", "operator_id": "11" * 32, "verdict": "match"},
        {"job_id": "j2", "operator_id": "22" * 32, "verdict": "mismatch"},
        {"job_id": "j3", "operator_id": "33" * 32, "verdict": "mismatch"},
        {"job_id": "j4", "operator_id": "11" * 32, "verdict": "mismatch"},
    ]
    faulty = faulty_operators_from_verdicts(receipts=[], verdicts=verdicts)
    assert faulty == {"22" * 32, "33" * 32, "11" * 32}


def test_faulty_operators_from_verdicts_is_case_insensitive_on_verdict() -> None:
    verdicts = [{"job_id": "j1", "operator_id": "11" * 32, "verdict": "Mismatch"}]
    faulty = faulty_operators_from_verdicts(receipts=[], verdicts=verdicts)
    assert faulty == {"11" * 32}


def test_faulty_operators_falls_back_to_receipt_operator_id() -> None:
    """When a verdict dict omits operator_id, resolve it from receipts by job_id."""
    from mining_types import Receipt

    r = Receipt(
        job_id="j9", operator_id="44" * 32, model_id="m",
        model_weight_hash="w", customer_nonce="n",
        request_hash="rq", response_hash="rs",
        kernel_pack_hash="k", attestation_report_hash="a",
        timestamp_ms=1, gateway_id="gw",
    )
    verdicts = [{"job_id": "j9", "operator_id": None, "verdict": "mismatch"}]
    faulty = faulty_operators_from_verdicts(receipts=[r], verdicts=verdicts)
    assert faulty == {"44" * 32}


# ------------------------------------------------------------------ app endpoint

def _hex_op(i: int) -> str:
    return f"{i:064x}"


def _config() -> object:
    from validator_replay import ValidatorConfig

    # 64-hex-char sr25519 seed placeholder; tests never actually sign.
    return ValidatorConfig(
        validator_id="val-1",
        validator_private_key_hex="11" * 32,
        sample_rate=0.5,
    )


def test_submit_weights_endpoint_computes_vector_and_calls_signer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The endpoint builds a vector from operator_stake, encodes the call
    body, and delegates signing+submission to submit_yuma_weights."""
    from fastapi.testclient import TestClient

    import validator_replay.app as appmod
    from validator_replay import build_app

    captured: dict[str, object] = {}

    def fake_submit(client, *, epoch, vector, validator_private_key_hex):
        captured["epoch"] = epoch
        captured["vector_len"] = len(vector.entries)
        captured["key_present"] = bool(validator_private_key_hex)
        return "0xdeadbeef"

    monkeypatch.setattr(appmod, "submit_yuma_weights", fake_submit)
    app = build_app(_config())  # type: ignore[arg-type]
    with TestClient(app) as c:
        r = c.post(
            "/submit_weights",
            json={
                "epoch": 7,
                "faulty_operators": [_hex_op(2)],
                "equal_weight": True,
                "operator_stake": {_hex_op(i): 100 for i in range(4)},
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["epoch"] == 7
    # 4 operators, one faulty -> 3 clean entries
    assert body["operator_count"] == 3
    assert body["tx_hash"] == "0xdeadbeef"
    assert body["submitted_to_chain"] is True
    assert captured["epoch"] == 7
    assert captured["vector_len"] == 3
    assert captured["key_present"] is True
    # call body carries the Yuma pallet index (13) + call index (0)
    assert body["call_body_hex"][:4] == "0d00"
    # epoch 7 little-endian follows
    assert body["call_body_hex"][4:20] == "0700000000000000"


def test_submit_weights_endpoint_400_without_validator_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No validator_private_key_hex -> 400, not a chain submission attempt."""
    from fastapi.testclient import TestClient

    from validator_replay import ValidatorConfig, build_app

    cfg = ValidatorConfig(
        validator_id="val-1", validator_private_key_hex="", sample_rate=0.5,
    )
    app = build_app(cfg)
    with TestClient(app) as c:
        r = c.post(
            "/submit_weights",
            json={"epoch": 1, "operator_stake": {_hex_op(0): 100}},
        )
    assert r.status_code == 400
    assert "private_key" in r.json()["detail"].lower()


def test_submit_weights_endpoint_returns_null_tx_hash_in_stub_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In stub mode (chain unreachable, VALIDATOR_ALLOW_STUB_CHAIN=1) the
    signer returns None and submitted_to_chain is False, but the computed
    vector + call body are still returned for inspection."""
    from fastapi.testclient import TestClient

    from validator_replay import build_app

    monkeypatch.setenv("VALIDATOR_ALLOW_STUB_CHAIN", "1")
    # Force the chain probe to fail so the stub path is taken for both the
    # operator fetch and the weight submission.
    import validator_replay.chain as chainmod

    monkeypatch.setattr(chainmod.ChainRpcClient, "_probe_with_retry", lambda self: False)
    app = build_app(_config())  # type: ignore[arg-type]
    with TestClient(app) as c:
        r = c.post("/submit_weights", json={"epoch": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    # get_operator_stakes fell back to the (empty) stub -> 0 operators
    assert body["operator_count"] == 0
    assert body["tx_hash"] is None
    assert body["submitted_to_chain"] is False
