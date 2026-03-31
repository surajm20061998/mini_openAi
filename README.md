### Download data
Download the TinyStories data:

``` sh
mkdir -p data
mkdir -p experiments
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

### Create smaller datasets for testing

```sh
python3 -c "from pathlib import Path; text=Path('data/TinyStoriesV2-GPT4-train.txt').read_text(encoding='utf-8', errors='ignore').split(); Path('data/train_500000w.txt').write_text(' '.join(text[:500000]), encoding='utf-8')"

python3 -c "from pathlib import Path; text=Path('data/TinyStoriesV2-GPT4-valid.txt').read_text(encoding='utf-8', errors='ignore').split(); Path('data/valid_500000w.txt').write_text(' '.join(text[:500000]), encoding='utf-8')"
```
### Running Tokenization Experiments
```sh

python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 1 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w1.npy \
  --out_val_npy data/val_tokens_500000w_w1.npy


python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 2 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w2.npy \
  --out_val_npy data/val_tokens_500000w_w2.npy


python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 4 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w4.npy \
  --out_val_npy data/val_tokens_500000w_w4.npy


  python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 8 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w8.npy \
  --out_val_npy data/val_tokens_500000w_w8.npy
```

### Running Tokenization Experiments on Full Dataset

```sh
python3 scripts/make_token_npy.py \
  --train_txt data/TinyStoriesV2-GPT4-train.txt \
  --val_txt data/TinyStoriesV2-GPT4-valid.txt \
  --vocab_size 512 \
  --num_workers 8 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_full_w8.npy \
  --out_val_npy data/val_tokens_full_w8.npy
```

### Experiment Records are Saved in /experiments

### Download Token Artifacts From Google Drive

```sh
python3 scripts/download_token_artifacts.py \
  --train "GOOGLE_DRIVE_TRAIN_URL_OR_FILE_ID" \
  --val "GOOGLE_DRIVE_VAL_URL_OR_FILE_ID" \
  --vocab "GOOGLE_DRIVE_VOCAB_URL_OR_FILE_ID" \
  --merges "GOOGLE_DRIVE_MERGES_URL_OR_FILE_ID"
```

This creates `data/` if needed and saves the downloaded files there. The Google Drive files must be shared as `Anyone with the link`.


