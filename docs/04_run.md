# 0) Install (once)
python -m pip install -e .[dev]

# 1) Quick QA (optional)
python -m tanailite.tools.run_smoke
python -m tanailite.tools.repro_mini_run

# Common paths
DATA=./data
TOKENIZER=./data/tokenizer
CORPUS=./data/corpus
ENCODER=./data/encoder
MODEL=./data/model
SFT=./data/sft/

mkdir -p $DATA/tokenizer $DATA/encoder $DATA/model $DATA/sft

# Optional control = PYTHONPATH=/home/lite

# 2) Tokenizer
python -m tanailite.train.train_tokenizer \
  --corpus-dir "$CORPUS" \
  --corpus-glob "*.txt" \
  --recursive \
  --model-prefix "$TOKENIZER/tanai-tokenizer" \
  --vocab-size 32000 \
  --model-type unigram \
  --character-coverage 0.9995 \
  --num-threads 16

# 3) Encoder
python -m tanailite.train.train_encoder \
  --tokenizer-model "$TOKENIZER/tanai-tokenizer.model" \
  --corpus-dir "$CORPUS" \
  --corpus-glob "*.txt" \
  --recursive \
  --max-seq-len 1024 \
  --d-model 512 \
  --n-layers 4 \
  --n-heads 8 \
  --ffn-dim 2048 \
  --out-dim 512 \
  --max-steps 3000 \
  --batch-size 32 \
  --eval-every 100 \
  --save-every 200 \
  --out-ckpt "$ENCODER/encoder_latest.pt" \
  --out-best "$ENCODER/encoder_best.pt" \
  --report-out "$ENCODER/encoder_report.json"

# 4) Base Model
python -m tanailite.train.train_base \
  --tokenizer-model "$TOKENIZER/tanai-tokenizer.model" \
  --train-data "$CORPUS" \
  --data-glob "*.txt" \
  --recursive \
  --max-seq-len 1024 \
  --d-model 512 \
  --n-layers 8 \
  --n-heads 8 \
  --mlp-ratio 4.0 \
  --tie-embeddings \
  --max-steps 5000 \
  --batch-size 8 \
  --eval-every 100 \
  --save-every 500 \
  --out-ckpt "$MODEL/base_latest.pt" \
  --out-best "$MODEL/base_best.pt" \
  --report-out "$MODEL/base_report.json"

# 5) SFT 
python -m tanailite.train.train_sft \
  --tokenizer-model "$TOKENIZER/tanai-tokenizer.model" \
  --base-ckpt "$MODEL/base_best.pt" \
  --sft-jsonl "$SFT/train.jsonl" \
  --template-name instruct \
  --max-prompt-tokens 512 \
  --max-response-tokens 384 \
  --max-steps 2000 \
  --batch-size 8 \
  --eval-every 100 \
  --save-every 200 \
  --out-ckpt "$SFT/sft_latest.pt" \
  --out-best "$SFT/sft_best.pt" \
  --report-out "$SFT/sft_report.json"

# 6) Inference
python -m tanailite.infer.run_infer \
  --tokenizer-model "$TOKENIZER/tanai-tokenizer.model" \
  --model-ckpt "$SFT/sft_best.pt" \
  --prompt "TanAILite nedir, kısa açıkla." \
  --prompt-template-name instruct \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --max-new-tokens 128 \
  --print-meta