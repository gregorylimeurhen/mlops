# SUTD404: Denoising Language Models for Room Address Lookup

**Web application**: [sutd404.vercel.app](https://sutd404.vercel.app).

## Setup

Follow these steps after cloning our repository if you want to run our code locally.
The latest weights are available in `./experiments/runs/`.

### Web Application

Follow these steps if you want to run our web application locally.

1. Optionally, configure `./experiments/config.toml`.
2. Optionally, run `./app/build.py`.
3. Open `./app/index.html`. For example:
```bash
# sutd404 $
            open ./app/index.html # MacOS
```

### Experiments

Follow these steps if you want to run our experiments locally.

1. Rename `./experiments/.env.example` to `./experiments/.env`. For example:
```bash
# sutd404 $
            cd experiments
            mv .env.example .env # MacOS
```
2. Set `WANDB_API_KEY` in `./experiments/.env` to working [Weights & Biases (W&B)](http://wandb.ai) API key.
3. Install packages in `./experiments/requirements.txt`. For example:
```bash
# sutd404 $
            cd experiments
            pip install -r requirements.txt
```
4. Install [torch](https://pytorch.org/get-started/locally).
5. Run pipeline. For example:
```bash
# sutd404 $
            cd experiments
            python -B preprocess.py # Preprocessing
            python -B train.py # Training
            python -B test.py # Testing
```

## Structure

```
.
├── app/
│   ├── assets.json
│   ├── build.py                 # build script, entry
│   ├── deploy.py                # deployment script, entry
│   ├── index.css
│   ├── index.html               # app markup, entry
│   ├── index.js
│   ├── weights.bin
│   └── worker.js
├── experiments/
│   ├── .env
│   ├── .env.example
│   ├── config.toml              # pipeline configuration, entry
│   ├── data/
│   │   ├── aliases.tsv          # room aliases
│   │   ├── boundaries.txt       # keyboard boundary list
│   │   ├── edges.tsv            # room-to-address relations
│   │   ├── layout.txt           # keyboard layout map
│   │   ├── n2a.tsv              # room-to-address function
│   │   ├── neighbors.json       # keyboard neighbour map
│   │   ├── test.tsv             # testing dataset
│   │   ├── train.tsv            # training dataset
│   │   └── val.tsv              # validation dataset
│   ├── preprocess.py            # preprocessing script
│   ├── requirements.txt
│   ├── runs/
│   │   └── 5327/
│   │       ├── test/
│   │       │   ├── results/
│   │       │   └── snapshot.zip
│   │       └── train/
│   │           ├── latest.pt    # latest weights
│   │           ├── model.pt     # best weights
│   │           ├── snapshot.zip
│   │           └── wandb/       # training logs
│   ├── test.py                  # testing script, entry
│   ├── train.py                 # training script, entry
│   └── utils.py
├── paper/
│   ├── bibliography.bib
│   ├── main.pdf
│   └── main.tex
├── slides/
│   ├── bibliography.bib
│   ├── main.pdf
│   └── main.tex
└── README.md
```
