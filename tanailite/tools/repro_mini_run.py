"""Deterministic mini-run check for TanAILite."""

import json
import torch
import torch.nn.functional as F

from tanailite.infer.generate  import TanAILiteGenerationConfig, generate_ids
from tanailite.model.tanai_gpt import TanAILiteGPT, TanAILiteGPTConfig
from tanailite.utils.runtime   import set_seed

# EN: These configuration values ​​are for testing purposes only. Do not confuse them with the actual configuration structure!
# TR: Bu Config değerleri sadece test içindir. Gerçek config yapısıyla karıştırmayın! 
def _run_once(seed: int) -> dict:
    set_seed(seed, deterministic=True)

    cfg = TanAILiteGPTConfig(
        vocab_size     = 96,
        d_model        = 64,
        n_layers       = 2,
        n_heads        = 4,
        max_seq_len    = 32,
        tie_embeddings = True,
        dropout        = 0.0,
    )
    model     = TanAILiteGPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Fixed synthetic batch so each run is byte-identical with the same seed.
    input_ids = torch.tensor(
        [
            [1, 7, 9, 12, 4, 19, 2, 0],
            [1, 8, 5, 21, 3, 11, 2, 0],
            [1, 3, 6, 14, 8, 16, 2, 0],
            [1, 10, 13, 4, 7, 15, 2, 0],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()

    model.train()
    logits = model(input_ids)
    loss_before = F.cross_entropy(logits[:, :-1, :].reshape(-1, cfg.vocab_size), labels[:, 1:].reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss_before.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_after = model(input_ids)
        loss_after = F.cross_entropy(
            logits_after[:, :-1, :].reshape(-1, cfg.vocab_size),
            labels[:, 1:].reshape(-1),
        )

    gen_cfg = TanAILiteGenerationConfig(max_new_tokens=12, temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
    output_ids, completion_ids, stop_reason = generate_ids(model, [1, 7, 9, 12], gen_cfg)

    return {
        "loss_before"    : float(loss_before.detach().item()),
        "loss_after"     : float(loss_after.detach().item()),
        "output_ids"     : [int(i) for i in output_ids],
        "completion_ids" : [int(i) for i in completion_ids],
        "stop_reason"    : str(stop_reason),
    }

def main() -> int:
    seed  = 2303
    run_a = _run_once(seed)
    run_b = _run_once(seed)

    if run_a != run_b:
        print("Deterministic mini-run FAILED.")
        print(json.dumps({"run_a": run_a, "run_b": run_b}, ensure_ascii=True, indent=2))
        return 2

    print("Deterministic mini-run passed.")
    print(json.dumps(run_a, ensure_ascii=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# Test için sadece Run edin. python -m tanailite.tools.repro_mini_run
