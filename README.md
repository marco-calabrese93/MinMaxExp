# Predictive MinMax Experimentation (Connect Four)

This repository contains the course project **Predictive MinMax Experimentation** for the Artificial Intelligence course (a.y. 2025/26, University of L’Aquila).

## Project idea

Classical **Minimax** becomes quickly impractical due to exponential tree growth.  
**Predictive Minimax** mitigates this by (1) truncating search and (2) replacing the leaf heuristic with a **learned evaluator**.

## What we implement

We study Predictive Minimax on **Connect Four** with:

- a truncated search controlled by **depth L** and **branching factor K**;
- a neural network evaluator \(H_true\) trained via **self-play** to predict final outcomes \(z ∈ {-1,0,+1}\);
- an iterative self-bootstrapping loop: better evaluator → better search → better training data.

## Key focus: scheduling (L, K)

A central contribution is the comparison of strategies that control **L(t)** and **K(t)** over training iterations under the **same computational budget**:

- **Strategy A**: fixed baseline (L=2, K=4)
- **Strategy D**: phased curriculum increasing depth/width over time
- **Strategy E**: rule-based curriculum with **K = 2L + 1**

## Evaluation protocol

Agents are evaluated with:

- **online gameplay metrics**: win/draw/loss rate vs random opponent and head-to-head matches;
- **offline predictive metrics**: MSE and Pearson correlation vs deeper Minimax estimates (L=4);
- **learning dynamics**: training loss and win-rate curves across iterations.

## Main results (from the report)

Curriculum-based schedules improve both stability and playing strength:

- offline MSE: A=0.2740, D=0.1743, E=0.1817
- correlation: A=0.0992, D=0.1749, E=0.1382
- win rate vs random (final): A=0.96, D=0.99, E=0.98  
  Overall, **Strategy D** performs best, while **Strategy E** offers a smoother and interpretable curriculum.

## Repository structure

- `game/` Connect Four environment (deterministic engine + API)
- `ai/` Minimax / Predictive Minimax + neural evaluator (`ai/model.py`)
- `experiments/` training/evaluation scripts (e.g. `train_predictive_minmax_D.py`)
- notebook analysis: plots, tables, and consolidated results

## Reproducibility

Experiments are designed to be reproducible via deterministic environment, fixed seeds, saved checkpoints, and CSV logs used by the analysis notebook.
