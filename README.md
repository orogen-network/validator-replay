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
scheme. Validators dogfood the network — running the same client other operators run is the cheapest way to keep us honest.

## Replay input privacy

Independent replay needs the original request input, not only the receipt hash.
`validator-replay` treats those `replay_inputs` as ephemeral sensitive data:

- `ephemeral` is the only supported production policy. In production, worker
  replay refuses to start unless `VALIDATOR_REPLAY_INPUT_POLICY=ephemeral` or
  the equivalent `ValidatorConfig.replay_input_policy` field is set.
- Inputs are accepted in the `/replay` or `/run_epoch` request body. Those
  replay request bodies are capped before JSON parsing by
  `replay_input_max_bytes` (default: 1 MiB), forwarded to the validator-owned
  worker, and then dropped with request-local memory.
- The service does not persist replay inputs, include them in API responses, or
  expose a retrieval endpoint for them.
- Validators that need longer retention must store encrypted inputs in their own
  audited evidence vault and submit only the needed replay batch to this service.

## Chain RPC

`chain_rpc_url` defaults to `ws://127.0.0.1:9944`. Override via the
`VALIDATOR_CHAIN_RPC_URL` env variable or by setting the field on
`ValidatorConfig` explicitly.

When the chain-node is not reachable, `ChainRpcClient` raises by default. Stub
operator data is available only when `VALIDATOR_ALLOW_STUB_CHAIN=1` is set.
Slashing evidence is still retained in memory (`app.state.slashing_submitted`)
so it can be resubmitted once the chain returns.
