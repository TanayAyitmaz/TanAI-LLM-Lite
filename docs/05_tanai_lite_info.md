Model:<br> 
**TanAILiteGPT**(<br>
&emsp;(tok_emb): Embedding(32000, 512)<br>
&emsp;(pos_emb): Embedding(1024, 512)<br>
&emsp;(blocks): ModuleList(<br>
&emsp;&emsp;(0-7): 8 x TanAILiteBlock(<br>
&emsp;&emsp;&emsp;(norm_attn): TanAIRMSNorm()<br>
&emsp;&emsp;&emsp;(q_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(k_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(v_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(o_proj): Linear(in_features=512, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(norm_mlp): TanAIRMSNorm()<br>
&emsp;&emsp;&emsp;(ff_up): Linear(in_features=512, out_features=2048, bias=False)<br>
&emsp;&emsp;&emsp;(ff_down): Linear(in_features=2048, out_features=512, bias=False)<br>
&emsp;&emsp;&emsp;(dropout): Dropout(p=0.0, inplace=False)))<br>
&emsp;(norm): TanAIRMSNorm()<br>
&emsp;(lm_head): Linear(in_features=512, out_features=32000, bias=False))<br>

Model Parameters *(tied)* : **42.08 M**
