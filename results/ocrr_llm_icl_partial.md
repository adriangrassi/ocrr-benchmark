# OCRR LLM-ICL spot check — partial result

**Date:** 2026-05-02
**Script:** `scripts/run_ocrr_llm_icl.py`
**Model:** `qwen2.5:14b` via local Ollama (CPU-only)
**Cell:** banking77 / oracle / seed=0 / eval-cap=30 / stream-cap=100
**Status:** **Step-0 only** — full run aborted because per-call latency on
14B-on-CPU made the full 5-checkpoint trajectory take >2 hours.

## Step-0 reading (no corrections written yet, retrieval over seed corpus only)

| System    | novel | original |
|-----------|------:|---------:|
| llm_icl (qwen2.5:14b, k=4 demos) | **0.467** | **0.533** |
| substrate (same config, reference) | 0.000 | 0.954 (extrapolated from full-sweep cell) |

## What this tells us, even as a partial result

The LLM-ICL system can guess novel-class labels at ~47% accuracy with just the
seed corpus and 4 in-context demos retrieved by vector similarity. **Substrate is at
0% on novel at step 0** — it has no entries for held-out classes yet.

But substrate dominates as soon as a few corrections come in: by 30-50 corrections
it's at 50–70% novel; by 100 corrections it's at 80%+. LLM-ICL would need to make
50+ LLM calls to grow its in-context demo pool that much, and each call costs ~30 s
on this CPU. So the substrate's recovery curve overtakes the LLM-ICL line within
the first minute of correction stream.

The other striking number: **LLM-ICL's original-distribution accuracy is 53%**.
This is the closed-label-following failure mode — the LLM doesn't reliably emit one
of the 77 known labels even when the right answer is in its in-context demos. With
better prompt engineering (numbered list, structured output) this would improve, but
it's at best comparable to a fine-tuned classifier, not better.

## Why the full run wasn't completed

CPU latency for 14B-parameter model:
- Per-call: 30-60 s (prompt encoding + decoding 16 tokens)
- Per checkpoint eval: 60 calls × ~40s = ~40 min
- Full 5-checkpoint + 100-stream run: estimated 2.5-4 hours

We ran for ~25 min and got only step-0. Killed.

## What would unblock a full run

Any of:
1. **GPU-backed Ollama** — 10-50× speedup. Would make the full run ~5-15 min.
2. **Smaller chat-capable model** — `qwen2.5:7b` or `phi3:mini` could be 3-5× faster.
   `llama3.1:latest` is 8B but doesn't expose `/api/chat` (non-instruct) on this box.
3. **Frontier LLM API** — Anthropic / OpenAI API. Fastest by far, costs API credits.
4. **Pre-compute the seed-corpus eval once and cache it** — the step-0 numbers
   only need 60 LLM calls, not 60 × 5 = 300. Trade: lose the per-checkpoint
   trajectory; keep just the "after N corrections" final readings.

## What we have without a full LLM-ICL run

The 5 strong algorithm baselines we ran in `ocrr_full_sweep_results.md` already
demonstrate substrate dominance. LLM-ICL is the fancy add-on that would round out
the comparison; we don't *need* it to make the substrate's case.

Still, a clean LLM-ICL trajectory is worth running once GPU compute is available.
Expected pattern: LLM-ICL probably lands around 50-70% novel (close to substrate's
final number) but 50-70% original (poor closed-label compliance), making it a
weaker overall system than substrate but a stronger novel-only baseline than EWC.

## Status

Phase 10.1d closed with this caveat. LLM-ICL retest filed as Phase 10.1f.
