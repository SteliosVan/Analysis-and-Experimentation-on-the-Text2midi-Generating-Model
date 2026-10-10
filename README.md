# Analysis and Experimentation on Text2midi

Course project at the **National Technical University of Athens (NTUA)**, 2025.

We study **text2midi**, a model that generates symbolic music (MIDI) directly from text captions, and run a set of controlled experiments on top of it: re-implementing its objective evaluation pipeline, genre-specific fine-tuning on rock music, partial unfreezing of the pretrained model, and two Mixture-of-Experts (MoE) variants.

**Authors:** Giannis Gousios, Angelos-Efstathios Karasavvidis, Eleni Papaioannou, Stelios Vantarakis
**Supervisors:** Prof. Alexandros Potamianos, Charis Papaioannou

> This is a student project run with limited compute (Google Colab Pro, Kaggle). Results are indicative, not conclusive. Caveats are listed explicitly below.

---

## Contents

- [Background: the text2midi model](#background-the-text2midi-model)
- [What we did](#what-we-did)
- [Evaluation metrics](#evaluation-metrics)
- [Results](#results)
- [Findings](#findings)
- [Limitations and problems encountered](#limitations-and-problems-encountered)
- [Future work](#future-work)
- [Data, tools and compute](#data-tools-and-compute)
- [References](#references)

---

## Background: the text2midi model

text2midi (Bhandari et al., *"Text2midi: Generating Symbolic Music from Captions"*) is an end-to-end model that turns a text description into a MIDI file.

- **Architecture:** transformer encoder–decoder. The encoder is a **frozen, pretrained FLAN-T5** that maps the caption to dense embeddings. These act as conditioning inputs, through cross-attention, to an **autoregressive Transformer decoder** that generates MIDI tokens. The decoder uses FlashAttention, a linear output layer over the MIDI vocabulary and a categorical cross-entropy loss.
- **Representation:** MIDI is tokenized with the **REMI+** tokenizer from the MidiTok library, which extends REMI with program-number and time-signature tokens, so multi-track, multi-instrument pieces with changing tempo are supported.
- **Training (original authors):** (1) semi-supervised pretraining on SymphonyNet (about 46k orchestral MIDI files) with pseudo-captions extracted by Music21; (2) supervised fine-tuning on **MidiCaps** (about 168k MIDI files with human-written captions), using sentence-omission regularization on the captions.

Our decoder-side setup: 18 Transformer layers, `d_model = 768`.

## What we did

### 1. Re-implementation of the objective evaluation pipeline
We implemented the evaluation protocol described in the original paper and ran it on **100 randomly selected samples from the MidiCaps test set**. The pipeline includes MIDI-to-audio conversion (for CLAP), key detection, tempo analysis and fallback mechanisms for robustness. One deliberate deviation: the original paper computes the compression ratio with COSIATEC; we use a lightweight approximation based on repeated pitch patterns (see [Limitations](#limitations-and-problems-encountered)).

### 2. Genre-specific fine-tuning (rock)
MidiCaps contains a large share of rock samples, so we asked whether decoder-only fine-tuning on rock data improves genre fidelity, rhythm and tonal correctness.

- **Data:** MidiCaps `train.json` filtered to entries whose genre field contains "rock" (about 33.8k samples). MIDI tokenized with REMI (max 512 tokens); captions encoded by the frozen FLAN-T5 encoder.
- **Trainable parameters:** decoder weights only (encoder frozen).
- **Hyperparameters:** AdamW, learning rate 2e-5, batch size 4, 3 epochs, single GPU.
- **Control:** a fine-tune on a *random* subset of MidiCaps, to separate the effect of "genre-specific data" from the effect of "any further fine-tuning".

### 3. Unfreezing and fine-tuning
Same rock data, but the **last 2 layers** of the original model are unfrozen and trained.

### 4. Mixture-of-Experts (two variants)
Both variants use 4 experts with a softmax gate, trained on 10,000 MidiCaps caption–MIDI pairs for 3 epochs (learning rate 1e-5 for the core layers, 2e-4 for the MoE block), with only the later Transformer layers and the MoE block trainable.

- **MoE v1:** the MoE block is applied *after* the Transformer output as a post-processing step; its output is added back to the hidden states through a residual connection before the final linear projection.
- **MoE v2:** same placement, but the MoE input is taken from **intermediate hidden states** captured with a forward hook on a LayerNorm module inside the Transformer, instead of the final output.

For both variants we inspected the softmax gate weights on a single test sample to qualitatively check how the experts are used.

## Evaluation metrics

| Metric | What it measures |
|---|---|
| **CR**: Compression Ratio | Repetition and structure; higher means more pattern reuse |
| **CLAP**: CLAP similarity | Alignment between the input text and the generated music (audio–text embeddings) |
| **TB**: Tempo Bin accuracy | Generated tempo falls in the same predefined bin as the reference |
| **TBT**: Tempo Bin with Tolerance | As TB, but neighbouring bins also count as correct |
| **CK**: Correct Key | Detected key exactly matches the reference |
| **CKD**: Correct Key with Duplicates | As CK, but relative keys (e.g. C major / A minor) are also accepted |

## Results

All values are averages over the evaluation samples. TB, TBT, CK and CKD are percentages.

| Metric | Paper (published) | Our re-evaluation of the base model | Random-subset fine-tune | **Rock fine-tune** | Rock + unfreeze last 2 layers ¹ | MoE v1 | MoE v2 |
|---|---|---|---|---|---|---|---|
| CR   | 2.31 ² | 1.24 | 0.75 | 0.92 | 1.37 | 0.00 | 0.11 |
| CLAP | 0.22 | 0.21 | 0.107 | 0.110 | 0.129 | 0.107 | 0.135 |
| TB   | 39.7 | 33 | 38 | 42 | 40 | 34 | 36 |
| TBT  | 65.8 | 66 | 72 | 70 | 66 | 61 | 72 |
| CK   | 33.6 | 34 | 28 | 33 | 14 | 12 | 18 |
| CKD  | 35.6 | 34 | 28 | 33 | 14 | 12 | 18 |

¹ Evaluated on **50** samples instead of 100, because of resource limits, so it is not directly comparable with the other columns.
² Computed with COSIATEC in the original paper. Our CR uses a cheaper approximation, so CR values are **not comparable** with this number. Compare CR only across our own runs.

<!-- TODO: verify the number of evaluation samples for the MoE v2 run and for the random-subset run and add it to the footnotes if different from 100. -->

### Gate usage on a single test sample

| Variant | Expert usage (4 experts) |
|---|---|
| MoE v1 | `[1.00, ~0, ~0, ~0]`: all probability mass on the first expert |
| MoE v2 | `[0.26, 0.04, 0.002, 0.69]`: two experts meaningfully active |

## Findings

- **The evaluation pipeline is a reasonable reproduction.** CLAP (0.21 vs 0.22), tempo-with-tolerance (66% vs 65.8%) and key accuracy (34% vs 33.6%) closely match the published values. Tempo-bin accuracy is lower (33% vs 39.7%), and CR differs because of the approximation described above.
- **Rock fine-tuning vs. a random-subset control:** the rock fine-tune scores higher on CR (0.92 vs 0.75), tempo-bin accuracy (42% vs 38%) and key accuracy (33% vs 28%), and is on par on CLAP. The random-subset control is slightly higher on TBT (72% vs 70%). Both fine-tunes score lower CLAP than the base model.
- **Unfreezing the last two layers** brings CR and CLAP closer to the base model than plain fine-tuning, but key accuracy drops to 14%. Because it was evaluated on half as many samples, we do not claim it is better or worse than the plain fine-tune.
- **MoE v1 did not help.** On the single sample we inspected, the gate collapsed onto one expert, and structure-sensitive metrics were poor (CR 0.00, CK 12%). Only tempo-with-tolerance stayed relatively stable, likely because tempo tokens are local and depend less on deep structure.
- **MoE v2 improved on v1** (CLAP 0.107 to 0.135, TBT 61% to 72%, CK 12% to 18%) and showed a more diverse gate, but still fell short of the base model. The gate analysis is based on one sample only and should be read as a qualitative hint of how the gate behaves, not as a measurement.

## Limitations and problems encountered

- **Compute:** training on Colab Pro (T4, L4, v6e-1 TPU) and Kaggle. Full decoder fine-tuning (18 layers, long sequences) is memory- and time-intensive; on CPU, batch sizes and throughput were severely constrained. This also limited the unfreezing run to 50 evaluation samples.
- **CR approximation:** see above. Absolute CR values are not comparable with the original paper.
- **Small MoE training budget:** 10,000 samples and 3 epochs, with the MoE block placed *outside* the core Transformer layers. Placing it inside earlier layers would require non-trivial architectural changes that the original codebase made impractical, and retraining from scratch was not feasible within the budget and timeline.
- **Positional encodings:** we kept the original rotary positional embeddings, because off-the-shelf implementations of music-aware attention or relative positional encodings were not available.
- **Swapping the text encoder fails.** We tried replacing the FLAN-T5 tokenizer and encoder with RoBERTa-base. Even though the tokenizer and encoder are consistent with each other, the decoder was trained on T5 encoder states and expects that format, so token IDs, vocabulary size (out-of-range indices) and cross-attention alignment break down. The decoder cannot be reused with a different encoder without retraining.

## Future work

- Mixed-precision **LoRA adapters** to cut the fine-tuning cost.
- **Music-aware relative positional encodings** to better capture long-range structure.
- Integrating the **MoE blocks inside earlier Transformer layers** so experts can influence the learned musical representations, trained on more data and for more epochs.
- Matching the original **COSIATEC** compression-ratio implementation so CR values can be compared with the paper.
- Evaluating all variants on the **same number of samples**, and gate usage over many samples rather than one.

## Data, tools and compute

- **Datasets:** SymphonyNet (about 46k symphonic multi-track MIDI files, used by the original authors for pretraining); MidiCaps (about 168k MIDI files with natural-language captions, derived from the Lakh MIDI Dataset), used for fine-tuning and evaluation.
- **Libraries:** Music21 (symbolic music feature extraction), MidiTok (REMI+ tokenization), CLAP (text–music alignment), FLAN-T5 as the text encoder.
- **Compute:** Google Colab Pro (T4, L4, v6e-1 TPU) and Kaggle.



## References

- Keshav Bhandari, Abhinaba Roy, Kyra Wang, Geeta Puri, Simon Colton, Dorien Herremans. *Text2midi: Generating Symbolic Music from Captions.*
- SymphonyNet dataset (Liu et al., 2022).
- MidiCaps dataset (Melechovsky et al., 2024), derived from the Lakh MIDI Dataset (Raffel, 2016).
