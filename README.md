# validator-replay

Stake-weighted, commit-reveal sampler that pulls a fraction of receipts per epoch,
replays inference against the validator's own worker pool, and emits slashing
extrinsics on mismatch (RFC-0005).

Sampling: Fisher-Yates shuffle weighted by operator stake, seeded by epoch hash
(per RFC-0006 placeholder). Until RFC-0006 is ratified this exposes a tunable
`sample_rate` parameter so the validator can be reused by chaos tests.

## Worker independence (operational rule)

**Validators MUST run their own worker pool**, completely independent of the
operator under audit. The `worker_replay_url` config field points at the
validator's local pool (e.g. an `infer-worker-vllm` deployment with the same
weights and kernel pack). Re-issuing the audited request to the *operator's*
worker would let a malicious operator cache the validator's deterministic
input and return the (already-tampered) answer — defeating the entire replay
scheme. See plan §0.3 ("validators dogfood the network").

## Chain RPC

`chain_rpc_url` defaults to `ws://127.0.0.1:9944`. Override via the
`VALIDATOR_CHAIN_RPC_URL` env variable or by setting the field on
`ValidatorConfig` explicitly.

When the chain-node is not reachable (probe times out within 100 ms),
`ChainRpcClient` falls back to a permanent on-disk stub under
`src/validator_replay/_stub_data/operators.json`. Slashing evidence is still
retained in memory (`app.state.slashing_submitted`) so it can be resubmitted
once the chain returns.
