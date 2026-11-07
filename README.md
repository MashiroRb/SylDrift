# SylDrift

Chinese zero-shot TTS with **one network evaluation**, trained by a
syllable-conditioned drifting objective.

## Idea

Drifting replaces iterative sampling with a distribution-matching field computed
during training. Applied naively to TTS it collapses: a sentence has exactly one
real recording, so the per-sentence positive pool has size one and the drift
degenerates into plain regression.

SylDrift moves the condition down to the **syllable**: Mandarin has ~1300 toned
syllables, each with thousands of real fragments in a few hundred hours of
speech. The drift field therefore sees a dense, well-estimated distribution.

Three components:

| Component | What it does | Failure it removes |
| --- | --- | --- |
| Syllable-conditioned drift (SCD) | drift field conditioned on toned syllable + local context | one-recording-per-sentence degeneracy |
| Timbre bypass | speaker prompt modulates every block via adaLN, never enters the drift field | distribution matching averaging out identity |
| Mel-domain feature space | frozen Audio-MAE on mel; no vocoder in the graph | non-differentiable waveform stage |

Boundaries come from the text side (duration predictor) and are turned into a
continuous frame-to-syllable assignment

    A[k,t] = sigmoid((t - b_{k-1})/tau) - sigmoid((t - b_k)/tau)

so the whole path stays differentiable. Outer boundaries are padded by 4*tau,
otherwise the first/last frames only receive half their weight (measured:
max column-sum deviation 0.5 -> 0.018).

## Status

- [x] Slurm environment (torch 2.5.1+cu121, conda env `syldrift`)
- [x] End-to-end smoke test on synthetic data (23.2M model, 1-NFE + drift + backward)
- [ ] AISHELL-3 download (in progress)
- [ ] Real-data training
- [ ] Evaluation (CER / SIM / UTMOS)

## Layout

```
syldrift/
  drift.py     soft assignment, pooling, drift field, loss
  model.py     small DiT flow model (text cross-attn + timbre adaLN)
  audio.py     wav -> log-mel
  data.py      manifest + dataset
  train.py     synthetic smoke
  train_real.py real training
slurm/         sbatch scripts
```
