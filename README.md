# TSP RL Comparison

A minimal, fair comparison framework for **RNN vs Transformer vs Linear Transformer** on Euclidean TSP with **REINFORCE**.

## Features
- Shared autoregressive pointer-style decoder
- Three interchangeable encoders:
  - `rnn`
  - `transformer`
  - `linear_transformer`
- Unified training setup:
  - REINFORCE with EMA baseline
  - entropy bonus
  - gradient clipping
- Logging of training dynamics:
  - training reward / tour length
  - validation tour length
  - entropy
  - gradient norm
  - reward variance
  - advantage variance
- Evaluation on:
  - uniform Euclidean TSP
  - clustered TSP
  - structured toy cases

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py --model transformer --n-cities 20 --steps 5000
python train.py --model rnn --n-cities 20 --steps 5000
python train.py --model linear_transformer --n-cities 20 --steps 5000
```

## Evaluate
```bash
python eval.py --ckpt checkpoints/transformer_step5000.pt --model transformer
python eval.py --ckpt checkpoints/rnn_step5000.pt --model rnn
python eval.py --ckpt checkpoints/linear_transformer_step5000.pt --model linear_transformer
```

## Plot training curves
```bash
python plot_logs.py --logdir logs
```

## Main files
- `data.py`: TSP instance generators
- `models.py`: shared decoder + 3 encoders
- `train.py`: RL training
- `eval.py`: evaluation across distributions
- `plot_logs.py`: training dynamics plots
- `config.py`: default hyperparameters

## Notes
- This is designed for course-project style **controlled comparison**, not SOTA TSP performance.
- The decoder is shared across models to improve fairness.
