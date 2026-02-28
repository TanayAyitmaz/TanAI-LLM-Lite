Model: TanAILiteGPT(
  (tok_emb): Embedding(32000, 512)
  (pos_emb): Embedding(1024, 512)
  (blocks): ModuleList(
    (0-7): 8 x TanAILiteBlock(
      (norm_attn): TanAIRMSNorm()
      (q_proj): Linear(in_features=512, out_features=512, bias=False)
      (k_proj): Linear(in_features=512, out_features=512, bias=False)
      (v_proj): Linear(in_features=512, out_features=512, bias=False)
      (o_proj): Linear(in_features=512, out_features=512, bias=False)
      (norm_mlp): TanAIRMSNorm()
      (ff_up): Linear(in_features=512, out_features=2048, bias=False)
      (ff_down): Linear(in_features=2048, out_features=512, bias=False)
      (dropout): Dropout(p=0.0, inplace=False)
    )
  )
  (norm): TanAIRMSNorm()
  (lm_head): Linear(in_features=512, out_features=32000, bias=False)
)
Model Params: 42.08 M