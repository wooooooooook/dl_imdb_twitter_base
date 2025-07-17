# 2025 DeepLearning

## cardiffnlp/twitter-roberta-base-sentiment-latest base fine tuned model

## How to use

### Download

```bash
git clone https://github.com/wooooooooook/imdb-sentiment-classifier.git imdb_model
cd imdb_model
git lfs pull
```

`git-lfs` needed

### Run

```bash
conda create -n imdb_py310 python=3.10 -y
conda activate imdb_py310

pip install torchtext==0.17.0 --force-reinstall \
            transformers==4.38.2 \
            accelerate==0.27.2 \
            torch \
            pandas \
            tqdm

python3 model.py <INPUT_DATA_PATH> model_weight
```

### Dataset download

```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

## Learning

- Use the `Twitter‑RoBERTa` tokenizer
- Set maximum input token length to `512`
- Configure batch size to `64`
- Freeze nine layers of the `Twitter‑RoBERTa` model and fine‑tune the remaining three layers
- Employ `GradScaler` with `autocast` for 16‑bit floating‑point precision
- Learning rates:
- RoBERTa layers: `2 × 10⁻⁵`
- Fully connected layer: `1 × 10⁻⁴`
- Apply a weight decay of `0.01`
- Save model weights each time a new best metric is achieved
- Training performed on a single `NVIDIA GeForce RTX 3090` GPU, taking approximately 20 minutes

## 제작

[wooooooooook](https://github.com/wooooooooook)
