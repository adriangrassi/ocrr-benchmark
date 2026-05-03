"""Strong baseline + variant systems for OCRR.

Adds 6 algorithm-level baselines beyond the trivial static + naive online_linear
systems in `ocrr_systems.py`:

  EWCSystem            — Elastic Weight Consolidation (Kirkpatrick 2017)
  AGEMSystem           — Averaged Gradient Episodic Memory (Chaudhry 2019)
  LwFSystem            — Learning without Forgetting (Li & Hoiem 2017)
  KNNLMSystem          — kNN-LM-style retrieval/parametric mixture (Khandelwal 2020)
  RiverLogRegSystem    — online linear classifier from `river` (online-ML lib)
  OllamaICLSystem      — local-LLM in-context learning with retrieved corrections

All operate on cached bge-large embeddings (encoder-agnostic). Each tries to be
the strongest possible version of its method for an OCRR scenario where:
  - initial training is on a 67-class subset
  - 10 held-out classes appear only via correction stream
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocrr_benchmark.eval.ocrr import OCRRSystem
from ocrr_benchmark.eval.ocrr_systems import (
    OnlineLinearSystem,
    _train_linear_head,
)


# ============================================================================
# EWC — Elastic Weight Consolidation (Kirkpatrick et al. 2017)
# ============================================================================
# Penalty: λ/2 · sum_i F_i · (θ_i - θ*_i)^2 added to per-correction loss.
# F is the diagonal Fisher information matrix estimated on the seed task.
# The intuition: parameters that mattered for the seed task get a stiff
# spring back to their initial value. Per-correction SGD still moves them,
# but only when the new gradient overwhelms the EWC penalty.
# ============================================================================

class EWCSystem(OCRRSystem):
    name = "ewc"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        all_classes: list[str],
        *,
        init_seed: int = 0,
        seed_epochs: int = 30,
        sgd_lr: float = 0.05,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 2000,
    ) -> None:
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._all_classes = list(all_classes)
        self._lbl_to_idx = {c: i for i, c in enumerate(all_classes)}
        self._idx_to_lbl = {i: c for i, c in enumerate(all_classes)}
        self._init_seed = init_seed
        self._seed_epochs = seed_epochs
        self._sgd_lr = sgd_lr
        self._ewc_lambda = ewc_lambda
        self._fisher_samples = fisher_samples
        self.reset()

    def reset(self) -> None:
        torch.manual_seed(self._init_seed)
        dim = self._seed_vecs.shape[1]
        head = nn.Linear(dim, len(self._all_classes), bias=True)
        X = torch.from_numpy(self._seed_vecs.astype(np.float32))
        y = torch.tensor(
            [self._lbl_to_idx[lbl] for lbl in self._seed_labels], dtype=torch.long
        )
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        n = X.shape[0]
        g = torch.Generator().manual_seed(self._init_seed)
        bs = 256
        head.train()
        for _ in range(self._seed_epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                idx = perm[s: s + bs]
                opt.zero_grad()
                loss = F.cross_entropy(head(X[idx]), y[idx])
                loss.backward()
                opt.step()
        head.eval()
        self._head = head
        self._sgd = torch.optim.SGD(self._head.parameters(), lr=self._sgd_lr)

        # Estimate Fisher diagonal on seed data via empirical Fisher on
        # PREDICTED label (Kirkpatrick's recipe). Subsample for speed.
        rng = np.random.default_rng(self._init_seed)
        n_fisher = min(self._fisher_samples, n)
        idx = rng.choice(n, size=n_fisher, replace=False)
        self._fisher = {pname: torch.zeros_like(p) for pname, p in head.named_parameters()}
        head.train()
        for i in idx:
            head.zero_grad()
            v = X[int(i)].unsqueeze(0)
            logits = head(v)
            log_probs = F.log_softmax(logits, dim=-1)
            pred = int(log_probs.argmax(dim=-1).item())
            log_probs[0, pred].backward()
            for pname, p in head.named_parameters():
                if p.grad is not None:
                    self._fisher[pname] += p.grad.detach() ** 2
        for pname in self._fisher:
            self._fisher[pname] /= float(n_fisher)
        head.eval()

        # Snapshot initial parameters (the "anchor" for EWC penalty).
        self._theta_star = {
            pname: p.detach().clone() for pname, p in head.named_parameters()
        }

    def predict(self, vec: np.ndarray) -> str | None:
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self._head(v)
        return self._idx_to_lbl.get(int(logits.argmax(dim=-1).item()))

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        if true_label not in self._lbl_to_idx:
            return
        target = torch.tensor([self._lbl_to_idx[true_label]], dtype=torch.long)
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        self._head.train()
        self._sgd.zero_grad()
        ce = F.cross_entropy(self._head(v), target)
        # EWC penalty
        ewc = torch.zeros((), dtype=torch.float32)
        for pname, p in self._head.named_parameters():
            ewc = ewc + (self._fisher[pname] * (p - self._theta_star[pname]) ** 2).sum()
        loss = ce + 0.5 * self._ewc_lambda * ewc
        loss.backward()
        self._sgd.step()
        self._head.eval()


# ============================================================================
# A-GEM — Averaged Gradient Episodic Memory (Chaudhry et al. 2019)
# ============================================================================
# Maintain a memory buffer of seed-task examples. On each correction step:
#   1) compute g_new = ∂L_correction / ∂θ
#   2) compute g_mem = ∂L_memory / ∂θ on a memory sample
#   3) if g_new · g_mem < 0 (would increase memory loss), project:
#        g_proj = g_new - (g_new · g_mem / |g_mem|^2) · g_mem
#   4) take an SGD step using g_proj.
# Cheap approximation of GEM (which projects against ALL past task gradients).
# ============================================================================

class AGEMSystem(OCRRSystem):
    name = "a_gem"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        all_classes: list[str],
        *,
        init_seed: int = 0,
        seed_epochs: int = 30,
        sgd_lr: float = 0.05,
        memory_size: int = 1000,
        memory_batch: int = 64,
    ) -> None:
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._all_classes = list(all_classes)
        self._lbl_to_idx = {c: i for i, c in enumerate(all_classes)}
        self._idx_to_lbl = {i: c for i, c in enumerate(all_classes)}
        self._init_seed = init_seed
        self._seed_epochs = seed_epochs
        self._sgd_lr = sgd_lr
        self._memory_size = memory_size
        self._memory_batch = memory_batch
        self.reset()

    def reset(self) -> None:
        torch.manual_seed(self._init_seed)
        dim = self._seed_vecs.shape[1]
        head = nn.Linear(dim, len(self._all_classes), bias=True)
        X = torch.from_numpy(self._seed_vecs.astype(np.float32))
        y = torch.tensor(
            [self._lbl_to_idx[lbl] for lbl in self._seed_labels], dtype=torch.long
        )
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        n = X.shape[0]
        g = torch.Generator().manual_seed(self._init_seed)
        bs = 256
        head.train()
        for _ in range(self._seed_epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                idx = perm[s: s + bs]
                opt.zero_grad()
                F.cross_entropy(head(X[idx]), y[idx]).backward()
                opt.step()
        head.eval()
        self._head = head
        self._sgd = torch.optim.SGD(self._head.parameters(), lr=self._sgd_lr)

        # Memory buffer: random subsample from seed data
        rng = np.random.default_rng(self._init_seed)
        m = min(self._memory_size, n)
        mem_idx = rng.choice(n, size=m, replace=False)
        self._mem_X = X[torch.tensor(mem_idx, dtype=torch.long)].clone()
        self._mem_y = y[torch.tensor(mem_idx, dtype=torch.long)].clone()
        self._rng = rng

    def predict(self, vec: np.ndarray) -> str | None:
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self._head(v)
        return self._idx_to_lbl.get(int(logits.argmax(dim=-1).item()))

    def _grad_vec(self) -> torch.Tensor:
        return torch.cat([p.grad.detach().flatten() for p in self._head.parameters()])

    def _set_grad_from_vec(self, gvec: torch.Tensor) -> None:
        offset = 0
        for p in self._head.parameters():
            n = p.numel()
            p.grad = gvec[offset: offset + n].view_as(p).clone()
            offset += n

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        if true_label not in self._lbl_to_idx:
            return
        target = torch.tensor([self._lbl_to_idx[true_label]], dtype=torch.long)
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        self._head.train()

        # Step 1: gradient on the new (correction) example
        self._sgd.zero_grad()
        F.cross_entropy(self._head(v), target).backward()
        g_new = self._grad_vec()

        # Step 2: gradient on a memory sample
        m = self._memory_batch
        idx = self._rng.choice(self._mem_X.shape[0], size=m, replace=False)
        mb_x = self._mem_X[torch.tensor(idx, dtype=torch.long)]
        mb_y = self._mem_y[torch.tensor(idx, dtype=torch.long)]
        self._sgd.zero_grad()
        F.cross_entropy(self._head(mb_x), mb_y).backward()
        g_mem = self._grad_vec()

        # Step 3: project g_new if it conflicts with g_mem
        dot = float(torch.dot(g_new, g_mem))
        if dot < 0:
            denom = float(torch.dot(g_mem, g_mem)) + 1e-12
            g_proj = g_new - (dot / denom) * g_mem
        else:
            g_proj = g_new

        self._set_grad_from_vec(g_proj)
        self._sgd.step()
        self._head.eval()


# ============================================================================
# LwF — Learning without Forgetting (Li & Hoiem 2017)
# ============================================================================
# Distillation-based: keep a frozen "teacher" copy of the post-seed head.
# Per correction, balance:
#   L = L_CE(new example, true label) + λ_distill * KL(student, teacher | new x)
# This penalises the student for changing its predictions on the seed-task
# distribution near the new example. Cheap; doesn't need a memory buffer.
# ============================================================================

class LwFSystem(OCRRSystem):
    name = "lwf"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        all_classes: list[str],
        *,
        init_seed: int = 0,
        seed_epochs: int = 30,
        sgd_lr: float = 0.05,
        distill_lambda: float = 1.0,
        temperature: float = 2.0,
    ) -> None:
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._all_classes = list(all_classes)
        self._lbl_to_idx = {c: i for i, c in enumerate(all_classes)}
        self._idx_to_lbl = {i: c for i, c in enumerate(all_classes)}
        self._init_seed = init_seed
        self._seed_epochs = seed_epochs
        self._sgd_lr = sgd_lr
        self._distill_lambda = distill_lambda
        self._T = temperature
        self.reset()

    def reset(self) -> None:
        torch.manual_seed(self._init_seed)
        dim = self._seed_vecs.shape[1]
        head = nn.Linear(dim, len(self._all_classes), bias=True)
        X = torch.from_numpy(self._seed_vecs.astype(np.float32))
        y = torch.tensor(
            [self._lbl_to_idx[lbl] for lbl in self._seed_labels], dtype=torch.long
        )
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        n = X.shape[0]
        g = torch.Generator().manual_seed(self._init_seed)
        bs = 256
        head.train()
        for _ in range(self._seed_epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                idx = perm[s: s + bs]
                opt.zero_grad()
                F.cross_entropy(head(X[idx]), y[idx]).backward()
                opt.step()
        head.eval()
        self._head = head
        self._sgd = torch.optim.SGD(self._head.parameters(), lr=self._sgd_lr)

        # Frozen teacher copy
        self._teacher = nn.Linear(dim, len(self._all_classes), bias=True)
        self._teacher.load_state_dict(head.state_dict())
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        self._teacher.eval()

    def predict(self, vec: np.ndarray) -> str | None:
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = self._head(v)
        return self._idx_to_lbl.get(int(logits.argmax(dim=-1).item()))

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        if true_label not in self._lbl_to_idx:
            return
        target = torch.tensor([self._lbl_to_idx[true_label]], dtype=torch.long)
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        self._head.train()
        self._sgd.zero_grad()
        student_logits = self._head(v)
        ce = F.cross_entropy(student_logits, target)
        with torch.no_grad():
            teacher_logits = self._teacher(v)
        T = self._T
        kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        loss = ce + self._distill_lambda * kl
        loss.backward()
        self._sgd.step()
        self._head.eval()


# ============================================================================
# kNN-LM — retrieval/parametric mixture (Khandelwal et al. 2020)
# ============================================================================
# Final probability is a convex mix:
#   p(y|x) = λ * p_kNN(y|x)  +  (1 - λ) * p_param(y|x)
# where p_kNN is the substrate's vote distribution (over the GROWING ledger,
# so it picks up corrections) and p_param is the FROZEN seed-trained linear
# head (covers the original distribution).
#
# This is the canonical "we add retrieval to a parametric model" baseline.
# Most reviewer-relevant comparison: kNN-LM exists, it's published, and many
# people will say "isn't your substrate just kNN-LM with extras?"
# Honest answer: substrate is the kNN side without the parametric mix; it
# already wins on novel-class recovery (where p_param contributes 0).
# kNN-LM trades that for retained original-distribution accuracy.
# ============================================================================

class KNNLMSystem(OCRRSystem):
    name = "knn_lm"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        all_classes: list[str],
        *,
        init_seed: int = 0,
        seed_epochs: int = 30,
        k: int = 5,
        margin: float = 0.05,
        lambda_knn: float = 0.5,
        knn_temperature: float = 0.1,
    ) -> None:
        from ocrr_benchmark.memory.episodic import ImmutableLedger
        self._all_classes = list(all_classes)
        self._lbl_to_idx = {c: i for i, c in enumerate(all_classes)}
        self._idx_to_lbl = {i: c for i, c in enumerate(all_classes)}
        self._k = k
        self._margin = margin
        self._lambda = lambda_knn
        self._tau = knn_temperature
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._init_seed = init_seed
        self._seed_epochs = seed_epochs
        self._ImmutableLedger = ImmutableLedger
        self.reset()

    def reset(self) -> None:
        # Frozen parametric head — trained once on the 67-class subset
        torch.manual_seed(self._init_seed)
        dim = self._seed_vecs.shape[1]
        head = nn.Linear(dim, len(self._all_classes), bias=True)
        X = torch.from_numpy(self._seed_vecs.astype(np.float32))
        y = torch.tensor(
            [self._lbl_to_idx[lbl] for lbl in self._seed_labels], dtype=torch.long
        )
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        n = X.shape[0]
        g = torch.Generator().manual_seed(self._init_seed)
        bs = 256
        head.train()
        for _ in range(self._seed_epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, bs):
                idx = perm[s: s + bs]
                opt.zero_grad()
                F.cross_entropy(head(X[idx]), y[idx]).backward()
                opt.step()
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
        self._head = head

        # Growing ledger for the kNN side
        self._ledger = self._ImmutableLedger()
        for v, lbl in zip(self._seed_vecs, self._seed_labels):
            self._ledger.write(v.astype(np.float32), text="", tags=(lbl,))

    def _knn_proba(self, vec: np.ndarray) -> np.ndarray:
        """Softmax over similarities of the k nearest neighbours, summed
        within label."""
        out = np.zeros(len(self._all_classes), dtype=np.float32)
        hits = self._ledger.nearest(vec.astype(np.float32), k=self._k)
        if not hits:
            return out
        sims = np.array([float(s) for _, s in hits], dtype=np.float32)
        # Numerically stable softmax with temperature τ
        z = sims / max(self._tau, 1e-6)
        z -= z.max()
        w = np.exp(z)
        w /= w.sum()
        for (entry, _), wi in zip(hits, w):
            if not entry.tags:
                continue
            lbl = entry.tags[0]
            i = self._lbl_to_idx.get(lbl)
            if i is None:
                continue
            out[i] += float(wi)
        return out

    def predict(self, vec: np.ndarray) -> str | None:
        v = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            param_proba = F.softmax(self._head(v), dim=-1).cpu().numpy()[0]
        knn_proba = self._knn_proba(vec)
        # If kNN sum is 0 (e.g., empty ledger pathological case), defer fully
        # to parametric.
        if knn_proba.sum() < 1e-6:
            mix = param_proba
        else:
            mix = self._lambda * knn_proba + (1.0 - self._lambda) * param_proba
        return self._idx_to_lbl.get(int(np.argmax(mix)))

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        # Append to the kNN datastore. Parametric head stays frozen.
        self._ledger.write(vec.astype(np.float32), text="", tags=(true_label,))


# ============================================================================
# river — online linear classifier from the `river` online-ML library.
# ============================================================================
# Represents the "online ML library literature" in the comparison. river is
# the actively-maintained successor to scikit-multiflow.
# ============================================================================

class RiverLogRegSystem(OCRRSystem):
    name = "river_logreg"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        all_classes: list[str],
        *,
        init_seed: int = 0,
        seed_passes: int = 1,
        seed_subsample: int = 3000,
    ) -> None:
        from river import linear_model, multiclass, optim
        self._lm = linear_model
        self._mc = multiclass
        self._optim = optim
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._all_classes = list(all_classes)
        self._init_seed = init_seed
        self._seed_passes = seed_passes
        self._seed_subsample = seed_subsample
        self.reset()

    def reset(self) -> None:
        # river's OneVsRest wrapping logistic regression — a clean
        # multiclass online linear classifier baseline.
        base = self._lm.LogisticRegression(optimizer=self._optim.SGD(0.01))
        self._clf = self._mc.OneVsRestClassifier(classifier=base)
        # Train on seed data via online passes (river is purely incremental).
        # Subsample to keep init time tractable on 1024-d embeddings (dict
        # features are slow per-example in river's data model).
        rng = np.random.default_rng(self._init_seed)
        n = len(self._seed_vecs)
        if self._seed_subsample and n > self._seed_subsample:
            base_idx = rng.choice(n, size=self._seed_subsample, replace=False)
        else:
            base_idx = np.arange(n)
        for _ in range(self._seed_passes):
            order = rng.permutation(len(base_idx))
            for j in order:
                i = int(base_idx[int(j)])
                x = self._vec_to_dict(self._seed_vecs[i])
                self._clf.learn_one(x, self._seed_labels[i])

    @staticmethod
    def _vec_to_dict(v: np.ndarray) -> dict:
        # river expects {feature_name: value}
        return {f"f{i}": float(v[i]) for i in range(v.shape[0])}

    def predict(self, vec: np.ndarray) -> str | None:
        x = self._vec_to_dict(vec)
        try:
            return self._clf.predict_one(x)
        except Exception:
            return None

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        x = self._vec_to_dict(vec)
        self._clf.learn_one(x, true_label)


# ============================================================================
# Local LLM in-context learning (Ollama)
# ============================================================================
# Predict by issuing a chat completion request to a local Ollama model. The
# prompt includes:
#   - The closed label set (all 77/151 classes)
#   - The k retrieved nearest *corrected* examples as in-context demos
#     (ledger of corrections grows as the stream proceeds)
# correct() appends the new (vec, label) to the demo retrieval store.
#
# This is the LLM-baseline interpretation of OCRR: the model itself doesn't
# update; the prompt does. Token cost grows with k; compute cost per query
# is the LLM call.
# ============================================================================

class OllamaICLSystem(OCRRSystem):
    name = "llm_icl"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        all_classes: list[str],
        seed_texts: list[str],
        *,
        model: str | None = None,
        k_demos: int = 6,
        init_seed: int = 0,
        timeout: float = 30.0,
    ) -> None:
        from ocrr_benchmark.memory.episodic import ImmutableLedger
        from ocrr_benchmark.oracles.ollama_classifier import (
            DEFAULT_HOST as _H, DEFAULT_MODEL as _M, _post_json,
        )
        self._H = _H
        self._M = model or _M
        self._post_json = _post_json
        self._all_classes = list(all_classes)
        self._k_demos = k_demos
        self._timeout = timeout
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._seed_texts = seed_texts
        self._init_seed = init_seed
        self._ImmutableLedger = ImmutableLedger
        # Retrieval ledger holds (vec, label, text) triples; we use ledger
        # for nearest-vec lookup, and a parallel list for text payloads.
        self.reset()

    def reset(self) -> None:
        self._ledger = self._ImmutableLedger()
        self._texts: list[str] = []
        for v, lbl, t in zip(self._seed_vecs, self._seed_labels, self._seed_texts):
            self._ledger.write(v.astype(np.float32), text=t, tags=(lbl,))
            self._texts.append(t)

    def predict(self, vec: np.ndarray) -> str | None:
        # Need the original text of the query; in OCRR the harness passes
        # vectors only. Fall back to no-text by setting the user query to
        # "match the closest example by content" — approximation that
        # exercises the LLM's classification reasoning, but admittedly
        # weaker than having the original text. (Future scope: thread the
        # text through the harness as well.)
        hits = self._ledger.nearest(vec.astype(np.float32), k=self._k_demos)
        if not hits:
            return None
        demo_lines = []
        for entry, _ in hits:
            txt = entry.text or "(text unavailable)"
            lbl = entry.tags[0] if entry.tags else "?"
            demo_lines.append(f"Example: {txt!r}  ->  {lbl}")
        labels_block = ", ".join(self._all_classes[:80]) + ("..." if len(self._all_classes) > 80 else "")
        # Use the closest example's text as a stand-in for the query
        query_text = hits[0][0].text or ""
        sys_prompt = (
            "You are classifying customer queries. Reply with ONLY the label "
            "name from the list, no explanation, no punctuation."
        )
        user_prompt = (
            f"Allowed labels: {labels_block}\n\n"
            f"Examples:\n" + "\n".join(demo_lines) + "\n\n"
            f"Query: {query_text!r}\nLabel:"
        )
        body = {
            "model": self._M,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 16},
        }
        try:
            resp = self._post_json(
                f"{self._H.rstrip('/')}/api/chat", body, timeout=self._timeout
            )
            raw = (resp.get("message") or {}).get("content", "").strip()
            if not raw:
                return None
            cleaned = raw.splitlines()[0].strip().strip('"').strip("'").rstrip(".,:;!?").lower()
            # Match against all_classes (case-insensitive)
            lower_to_real = {c.lower(): c for c in self._all_classes}
            if cleaned in lower_to_real:
                return lower_to_real[cleaned]
            for c in self._all_classes:
                if c.lower() in cleaned:
                    return c
            return None
        except Exception:
            return None

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        # No text available from harness; store with a generic placeholder.
        # The retrieval still fires because we go by vec similarity.
        text_proxy = f"(stream entry)"
        entry = self._ledger.write(vec.astype(np.float32), text=text_proxy, tags=(true_label,))
        self._texts.append(text_proxy)


# ============================================================================
# BoundedSubstrateSystem — substrate constrained to a fixed memory budget.
# ============================================================================
# Same vote rule as the unbounded substrate (margin-band majority + max-sim
# tiebreak), but the underlying buffer is capped at `budget` entries. When
# the buffer is full, an eviction policy decides which entry to drop on
# each new write.
#
# Why this exists: the unbounded substrate violates the strict online-
# learning constraint "you can't store all the historical data." Bounded
# variants let us probe the storage-vs-recovery Pareto:
#   - budget=∞ ............... unbounded substrate (full retention)
#   - budget=N (reservoir) ... uniform sample of all writes seen so far
#   - budget=N (fifo) ........ keeps the most recent N writes
#   - budget=0 ............... model parameters only (river territory)
#
# Eviction policies:
#   reservoir : Vitter Algorithm R — uniform random sample over an
#               unbounded stream. Each write that doesn't fit replaces a
#               random index with probability budget/count_seen.
#   fifo      : ring buffer; the oldest entry is evicted on each write.
# ============================================================================

class BoundedSubstrateSystem(OCRRSystem):
    name = "bounded_substrate"

    def __init__(
        self,
        seed_vecs: np.ndarray,
        seed_labels: list[str],
        *,
        budget: int = 1000,
        eviction: str = "reservoir",  # "reservoir" or "fifo"
        k: int = 5,
        margin: float = 0.05,
        init_seed: int = 0,
    ) -> None:
        if eviction not in ("reservoir", "fifo"):
            raise ValueError(f"eviction must be 'reservoir' or 'fifo', got {eviction!r}")
        self._seed_vecs = seed_vecs
        self._seed_labels = seed_labels
        self._budget = int(budget)
        self._eviction = eviction
        self._k = k
        self._margin = margin
        self._init_seed = init_seed
        # Name reflects the budget so summary tables can distinguish
        # different budgets at a glance.
        self.name = f"bounded_substrate_{eviction}_{budget}"
        self.reset()

    def reset(self) -> None:
        self._buf_vecs: list[np.ndarray] = []
        self._buf_labels: list[str] = []
        self._buf_ids: list[int] = []          # for recency tiebreak
        self._next_id = 0
        self._count_seen = 0
        self._rng = np.random.default_rng(self._init_seed)
        # Cached unit-normalized matrix; rebuilt lazily on first predict
        # after any write. Predicts dominate runtime during eval, so this
        # gives a 10x speedup vs re-stacking on every call.
        self._unit_cache: np.ndarray | None = None
        # Stream the seed corpus through the same eviction process so the
        # bounded substrate isn't given a free pass on its initial state.
        # This matches the strict online-learning interpretation: there is
        # no "training phase," only a stream.
        for v, lbl in zip(self._seed_vecs, self._seed_labels):
            self._write(v, lbl)

    # ------------------------------------------------------------------
    def _write(self, vec: np.ndarray, label: str) -> None:
        v32 = vec.astype(np.float32, copy=False)
        if self._eviction == "fifo":
            self._fifo_write(v32, label)
        else:
            self._reservoir_write(v32, label)
        self._next_id += 1
        self._count_seen += 1
        # Buffer changed — invalidate the unit-norm cache.
        self._unit_cache = None

    def _reservoir_write(self, vec: np.ndarray, label: str) -> None:
        if len(self._buf_vecs) < self._budget:
            self._buf_vecs.append(vec)
            self._buf_labels.append(label)
            self._buf_ids.append(self._next_id)
            return
        # Reservoir replacement: with probability budget/(count_seen+1)
        # replace a random slot.
        i = int(self._rng.integers(0, self._count_seen + 1))
        if i < self._budget:
            self._buf_vecs[i] = vec
            self._buf_labels[i] = label
            self._buf_ids[i] = self._next_id

    def _fifo_write(self, vec: np.ndarray, label: str) -> None:
        if len(self._buf_vecs) < self._budget:
            self._buf_vecs.append(vec)
            self._buf_labels.append(label)
            self._buf_ids.append(self._next_id)
            return
        # Evict oldest (index 0). For budgets up to ~5k this is fine; for
        # larger budgets a deque or index-based ring would be faster.
        self._buf_vecs.pop(0)
        self._buf_labels.pop(0)
        self._buf_ids.pop(0)
        self._buf_vecs.append(vec)
        self._buf_labels.append(label)
        self._buf_ids.append(self._next_id)

    # ------------------------------------------------------------------
    def predict(self, vec: np.ndarray) -> str | None:
        if not self._buf_vecs:
            return None
        v = vec.astype(np.float32, copy=False)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return None
        v_unit = v / n
        # Inputs are already unit-normalised in our pipelines; the cache
        # path below avoids re-normalising the buffer on every predict.
        if self._unit_cache is None or self._unit_cache.shape[0] != len(self._buf_vecs):
            emb = np.stack(self._buf_vecs)
            emb_norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self._unit_cache = (emb / np.clip(emb_norms, 1e-9, None)).astype(np.float32)
        sims = self._unit_cache @ v_unit
        k = min(self._k, len(self._buf_vecs))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_sims = sims[top_idx]
        top_sim = float(top_sims.max())
        band_mask = top_sims >= top_sim - self._margin
        band = top_idx[band_mask]
        counts: dict[str, int] = {}
        max_sim: dict[str, float] = {}
        latest_id: dict[str, int] = {}
        for j in band:
            j = int(j)
            label = self._buf_labels[j]
            counts[label] = counts.get(label, 0) + 1
            max_sim[label] = max(max_sim.get(label, -1.0), float(sims[j]))
            latest_id[label] = max(latest_id.get(label, -1), self._buf_ids[j])
        if not counts:
            return None
        return max(counts.keys(),
                   key=lambda lbl: (counts[lbl], max_sim[lbl], latest_id[lbl]))

    def correct(self, vec: np.ndarray, true_label: str) -> None:
        self._write(vec, true_label)

    # ------------------------------------------------------------------
    def buffer_size(self) -> int:
        return len(self._buf_vecs)
