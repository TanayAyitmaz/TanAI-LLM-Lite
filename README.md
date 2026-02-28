# TanAILite

## What is TanaAI and TanAI-Lite
**TanAI-Lite is the open-source version of the TanAI architecture. It is an open-source release of TanAI, simplified to be a GPT version of the actual model structure.**
TanAI utilizes many modern structures and has Fused, Ecv, and Chronos projections in the Transformer core. 
- **Fused**: 256D vector projection for semantic context consistency.
- **Ecv** *(emotional conditioning vector)*: 64D vector projection for emotional context consistency. Robert Plutchik's 8 emotion structure was used for Ecv, and thousands of emotional sentences were converted into vectors. A 48D vector is created from this dataset, and a 16D vector is created from the Emotional User Profile.
- **Chronos**: 32D vector projection for learning time frequencies. LLMs do not know time series and cannot learn from external prompts. Chronos was designed to perceive the past and predict the future.

SwiGLU was used as the activation function in TanAI. GeLU was used for Tanai-Lite.<br> 
AdaRMSNorm and ada_proj were used for normalization in TanAI. RMSNorm was used for Tanai-Lite.<br> 
TanAI is a modular LLM, not monolithic, and all decisions are observable with **GlassBox - Telemetry**.<br>
Lite is a version where many extra features have been simplified and released as open-source.<br>

## TanAI-Lite Config
Minimal open-source training and inference stack inspired by TanAI.

 Model: TanAILiteGPT(<br>
 _   (tok_emb): Embedding(32000, 512)<br>
  _  (pos_emb): Embedding(1024, 512)<br>
   _ (blocks): ModuleList(<br>
    _    (0-7): 8 x TanAILiteBlock(<br>
     _       (norm_attn): TanAIRMSNorm()<br>
            (q_proj): Linear(in_features=512, out_features=512, bias=False)<br>
            (k_proj): Linear(in_features=512, out_features=512, bias=False)<br>
            (v_proj): Linear(in_features=512, out_features=512, bias=False)<br>
            (o_proj): Linear(in_features=512, out_features=512, bias=False)<br>
            (norm_mlp): TanAIRMSNorm()<br>
            (ff_up): Linear(in_features=512, out_features=2048, bias=False)<br>
            (ff_down): Linear(in_features=2048, out_features=512, bias=False)<br>
            (dropout): Dropout(p=0.0, inplace=False)<br>
        )<br>
    )<br>
    (norm): TanAIRMSNorm()<br>
    (lm_head): Linear(in_features=512, out_features=32000, bias=False)<br>
)<br>

## Model parameters *(tied)*
Model Params: 42.08 M

## Hardware Feasibility *(30M-50M)*
The TanAI-Lite Model has ~42,082,816M parameters. This parameter structure can be modified with TanAILiteConfig. Training was performed smoothly with 16GB VRAM in current tests. Training can be performed with 12GB VRAM using lower training parameters (low Batch-Size).
Recommended:
- **3090 24GB** *(good training output)*
- **4090 24GB** *(good training output)*
- **5070TI 16GB** *(standard)*
- **5080 16GB** *(standard)*
- **5090 32GB** *(much better training output)*

Yes, this setup is trainable on single GPU with RTX 3090/4090/5090 class cards.
- 24GB VRAM *(3090/4090)*: comfortable for 30M-50M with AdamW + mixed precision.
- 16GB VRAM *(some 50xx SKUs)*: still workable with lower batch + grad accumulation.
- Main pressure is activation memory (sequence length), not raw parameter count.

Practical guidance:
- `seq_len=1024`: easy
- `seq_len=2048`: still practical
- `seq_len=4096`: possible but requires smaller micro-batch

## Scope
- Full open-source tokenizer / encoder / transformer
- Tokenizer training/eval
- Encoder training/eval
- Lite GPT model training
- SFT training
- Single-command inference

## Core CLIs
Download the Corpus in your desired language via HF and split the Corpus for testing.
- `tanailite-corpus-slicer`
Train a 32k Vocab Tokenizer with the Corpus dataset.
- `tanailite-train-tokenizer`
Train the Encoder with your tokenizer and corpus dataset. (For RAG and embedding vector generation)
- `tanailite-train-encoder`
You can fully train your model. (The base model is trained for 5k steps in the test command; you can achieve much better results by extending this training.)
- `tanailite-train-base`
Give your model personality. Download an instruction SFT dataset via HF and train your model.
- `tanailite-train-sft`
Perform the model's inference tests.
- `tanailite-infer`

## How do I run it?
Development environment on Python 3.10 and above.
You can follow the instructions and commands in the **docs/04_run.md** file.

## Base Model and Encoder Files
Base Model file: https://tanai.xyz/tanai/base_best.pt
The base model has only been trained on 5000 steps and has not yet learned the language. Please train it on at least 80-100k steps and perform inference checks.

Encoder file: https://tanai.xyz/tanai/encoder_best.pt
We recommend using encoder outputs that exceed values such as retrieval_at1 > 0.7, mrr > 0.50, mean_margin > 0.05 in the encoder reports. (This encoder file was trained for 300 steps for testing.)

## Reports
> [!NOTE]
> You can review the JSON files in the **data/reports** folder for the training reports.

## Docs
- 00_scope.md
- 01_architecture.md
- 02_training_flow.md
- 03_inference_flow.md
- 04_run.md
- 05_tanai_lite_info.md
- 06_corpus_selection.md