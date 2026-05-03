# Datasheet — OCRR streaming protocols

OCRR does not introduce a new dataset. It defines a **streaming protocol**
on top of two existing public datasets (Banking77 and CLINC150) and
documents the protocol here in the Gebru-format datasheet structure
(Gebru et al. 2018). For the upstream datasets themselves, see their
respective publications.

## Motivation

### Why was OCRR created?

To measure a property — recovery from distribution shift via online
correction — that static benchmarks (Banking77, GLUE, MMLU) cannot
characterise by construction. Static benchmarks freeze the model at
training time; OCRR scores systems on how *fast* they regain accuracy
when the test distribution shifts and corrections arrive online.

### Who created it?

The streaming protocol is an artifact of this paper's authors. The
underlying datasets (Banking77, CLINC150) come from the cited works.

### Was funding involved?

No external funding. The work was performed on personal hardware in
personal time.

## Composition

### Stream construction

For each (dataset × seed) pair, the stream is constructed as follows:

1. **Held-out classes:** 10 classes are randomly held out from the
   dataset's full label set (67 of 77 retained for Banking77; 140 of 150
   retained for CLINC150). Held-out class identities are seeded — same
   seed → same held-out set.
2. **Initial state:** systems are initialised on the train portion of
   the *retained* classes only.
3. **Stream:** test queries are then drawn — alternating in a fixed
   schedule — from the held-out (novel) and retained (original) portions
   of the test split. The held-out queries' true labels are unknown to
   the system at start.
4. **Correction:** when the system predicts wrong on a held-out query,
   the correction policy decides whether to reveal the true label. The
   system is then expected to update.

Stream lengths: ~4 000 items (Banking77), ~6 000 items (CLINC150).

### Eval sets

Two held-out evaluation sets are constructed once and never modified:

- **`novel`** — disjoint test queries from the 10 held-out classes
- **`original`** — test queries from the 67 retained classes

Both are used to compute accuracy at every checkpoint (default: every 50
stream steps).

## Collection

### How was data collected?

Banking77 and CLINC150 were collected via crowdsourcing by their
respective authors. OCRR adds no data — the protocol just imposes a
stream order, eval-set split, and correction-policy decision at each
step.

### Who labelled the data?

Crowdworkers, per the upstream datasets. Held-out classes are labelled
with the same labels as the original dataset; correction-on-stream
provides the same gold label that already exists in the dataset.

### Confidentiality

No new personal information is introduced by OCRR. Both Banking77 and
CLINC150 are public, anonymised intent-classification datasets.

## Preprocessing

Text is left unchanged. Encoders embed it directly:
- `BAAI/bge-large-en-v1.5` (1024-d) — primary encoder
- `BAAI/bge-small-en-v1.5` (384-d) — used for substrate variants

For the LoRA-DeBERTa baseline only, text is tokenized with
`microsoft/deberta-v3-large`'s tokenizer to a max length of 64.

## Uses

### What can OCRR be used for?

- Comparing how fast continual-learning / online-learning systems recover
  from class-incremental shift on canonical NLP benchmarks
- Characterising storage-vs-recovery trade-offs (the bounded-substrate
  Pareto)
- Stress-testing parameter-efficient fine-tuning approaches under
  per-correction supervision

### What should OCRR NOT be used for?

- Static accuracy comparisons on Banking77 or CLINC150 — use the
  upstream test split for that, not the OCRR stream.
- Out-of-distribution detection — the held-out classes are *new*, not
  *out-of-scope*. CLINC150-plus has its own OOS protocol for that.
- Intent classification under continuous concept drift — OCRR's shift is
  categorical (new classes appear), not within-class drift. The
  adversarial-paraphrase scenario in §3.4 of the paper sketches the
  within-class case as future work.

## Distribution

OCRR is distributed as code (this repository) under MIT. The streaming
protocol can be reproduced from the public datasets by running the
scripts in `scripts/`.

## Maintenance

The benchmark is maintained at
`https://github.com/adriangrassi/ocrr-benchmark`. Updates to baselines
or new system families will be added under tagged releases. The streaming
protocol itself (held-out class selection, stream order, eval-set
construction) is **frozen** for v1 to preserve cross-paper comparability.

If a future protocol revision is needed (e.g., a within-class drift
scenario), it will be released as `v2` and both versions will continue
to be supported.

## Ethical considerations

### Privacy

No new PII is introduced. The upstream datasets contain anonymised
banking and customer-service intent queries.

### Bias

The streaming protocol does not modify class labels, but it does
**select** which 10 classes are held out per seed. This selection is
random, but on Banking77 the randomly held-out set may include
demographically-skewed intents (e.g., specific country-of-origin
remittance flows). Aggregate metrics across multiple seeds (we report 3)
mitigate any single seed's selection bias.

### Misuse

The benchmark itself does not enable any new misuse vector beyond what
the upstream datasets already permit.

## References

- Casanueva, I. et al. (2020). Efficient intent detection with dual
  sentence encoders. NLP4ConvAI.
- Larson, S. et al. (2019). An evaluation dataset for intent
  classification and out-of-scope prediction. EMNLP.
- Gebru, T. et al. (2018). Datasheets for datasets. Communications of
  the ACM.
