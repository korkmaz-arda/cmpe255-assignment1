# Project Specification — NanoLlama: From-Scratch Transformer LLM & Interpretability Studio

Build-facing specification derived from static inspection of the source project. It describes required behavior for an independent reimplementation; it is not a transcription of the original code.

---

## 1. Project Overview

A single-page web application built around a small decoder-only language model that is written from scratch (no pretrained weights, no model library) and trained locally on a hand-authored instruction-tuning corpus. The application is equally a **chatbot** and a **transformer interpretability studio**: five tabs let a visitor talk to the model, inspect its attention matrices, decompose text into tokens, review the training run, and read the architecture blueprint. A modal lets the visitor retrain the model from the browser.

The data-science identity is **supervised fine-tuning (SFT) of a character-level causal language model using modern LLaMA-family primitives** — rotary position embeddings, SwiGLU gated feed-forward blocks, RMS normalization, weight-tied LM head, and an incremental key/value cache — with loss masked to the assistant span only. Model size is deliberately tiny (~0.5M parameters, ~100-token vocabulary) so the whole train/serve loop runs on a laptop CPU in minutes.

Critically: **the chat path is retrieval-first.** For most user messages the served reply is a verbatim canonical answer looked up from the training corpus and emitted character-by-character with an artificial delay, not text sampled from the network (see §6 and §7).

---

## 2. Required Features

### Core Requirements

---

**`F01` — Character-level tokenizer with chat control tokens**

**Behavior:** Text is encoded to integer ids using a fixed, deterministic, 1:1 character vocabulary plus a small set of reserved control tokens. No corpus-derived merge learning takes place; the vocabulary is constructed the same way every run.

**Important details:**

* Six reserved control ids occupy the lowest slots, in this fixed order: padding, beginning-of-sequence, end-of-sequence, user-turn marker, assistant-turn marker, system-turn marker. Their ids are surfaced in the UI (§F07) so the ordering is part of the user-facing contract.
* The remaining vocabulary is one id per printable ASCII character (space through tilde) plus newline, tab and carriage return. Total vocabulary ≈ 104 entries.
* Encoding scans left-to-right, preferring a control-token literal (e.g. the `<|user|>` marker written inline in a prompt string) over its constituent characters; unknown characters fall back to the space id.
* Decoding is the inverse map, with an option to suppress control tokens.
* A chat-formatting helper wraps a system prompt and a user message into a single prompt string terminated by the assistant marker, which is the canonical prompt shape used by both training and inference.
* The vocabulary is persisted alongside the model checkpoint and reloaded at serve time.

---

**`F02` — Hand-authored instruction corpus with paraphrase and persona expansion**

**Behavior:** The training corpus is defined in code, not loaded from any external dataset. It is a curated knowledge base of question→answer pairs, expanded combinatorially into chat dialogues.

**Important details:**

* ~44 knowledge entries. Each entry carries (a) a list of paraphrased user queries that should map to it, (b) one canonical assistant answer, (c) the subset of personas for which the entry is in-scope.
* Topic coverage: model identity and its own architecture (RoPE, SwiGLU, RMSNorm, KV cache); ML foundations (supervised/unsupervised, backprop, gradient descent, over/underfitting); evaluation metrics (precision/recall, F1, confusion matrix, cross-validation); classical algorithms (Naive Bayes, random forest, K-means, PCA, transformers, self-attention); short Python snippets (reverse string, Fibonacci, factorial, primality, binary search, palindrome); creative writing (bedtime story, Mars story); arithmetic/algebra word problems; and small talk (greetings, a joke, a fun fact).
* Six persona system prompts: generic assistant, "brilliant" assistant, ML research assistant, Python engineer, bedtime storyteller, math tutor.
* Expansion: every (query × in-scope persona) pair becomes a dialogue, and every query additionally gets one dialogue under the default persona. ~174 distinct queries expand to ~695 dialogues.
* This corpus is the single source of truth for both training targets and the served canonical answers (§F05), so the two must stay consistent.

---

**`F03` — SFT sequence construction with assistant-only loss masking**

**Behavior:** Each dialogue becomes one fixed-length next-token-prediction example in which only the assistant response contributes to the loss.

**Important details:**

* Sequence = BOS + formatted prompt (system + user + assistant marker) + assistant answer + EOS.
* Padded to a fixed context length (192 in the training entry point); longer sequences are truncated from the right, which silently drops the tail of the longest answers.
* Inputs and targets are the sequence shifted by one.
* Target positions covering the prompt span and all padding positions are set to an ignore sentinel so cross-entropy skips them. Loss is therefore computed only over assistant-response characters and the terminating EOS.
* The whole dialogue list is repeated a small number of times (2 in the training entry point) to lengthen an epoch.

---

**`F04` — Decoder-only transformer implemented from primitives**

**Behavior:** The network is built from explicitly implemented components; none of its distinguishing primitives may be swapped for plain alternatives, because the entire project exists to exhibit them and the architecture tab documents them to the user.

**Important details:**

* **Configuration as trained:** 3 layers, model width 128, 4 heads (head dim 32), SwiGLU inner width 256, context 192, vocabulary ≈104, ≈505,728 trainable parameters. (The model class carries larger defaults — 4 layers / width 192 / 6 heads / context 256 — which are *not* what the training entry point uses; documentation quotes several other numbers, see §7.)
* **RMSNorm** pre-normalization before both sub-blocks and once before the output projection: rescale by the root-mean-square of the feature vector (no mean subtraction) with a small epsilon, times a learned per-channel gain. Normalization is computed in float32 and cast back.
* **Rotary position embeddings** applied to queries and keys only (not values), from precomputed cosine/sine tables over positions × head-dim with the standard geometric inverse-frequency schedule (base 10000). Pair-wise rotation operates on *adjacent* coordinate pairs (interleaved layout), not on split halves — the frequency table is built by repeat-interleave to match. During cached decoding the table is sliced at the absolute position of the new token, so relative distances stay correct across steps.
* **Causal multi-head self-attention** with separate bias-free query/key/value/output projections, scaled dot-product scores, additive upper-triangular `-inf` mask. When decoding with a cache, the mask is left-padded with zeros so the new token may attend to all cached positions.
* **SwiGLU feed-forward**: two parallel bias-free up-projections, one passed through SiLU, multiplied elementwise, then a bias-free down-projection.
* Residual connections around both sub-blocks; final RMSNorm; linear LM head whose weights are **tied to the token embedding matrix**.
* Two attention execution paths exist for the same math: a fused kernel used for masked full-sequence forward passes, and an explicit score/softmax path used whenever the caller wants the cache or the attention matrix back. Both must produce equivalent outputs; the explicit path is what makes `F06` possible.

---

**`F05` — Train the model and persist a reusable checkpoint**

**Behavior:** A training run fits the model on the SFT corpus, reports per-epoch progress, and writes three artifacts that the serving path consumes: model weights + config, the tokenizer vocabulary, and a training-telemetry record.

**Important details:**

* Split: ~92% train / ~8% validation by random partition of the (already duplicated) example list. **This does not produce a held-out set** — see §7.
* Optimizer AdamW (betas 0.9/0.95, weight decay 0.01), gradient-norm clipping at 1.0.
* Learning-rate schedule: linear warmup over the first ~5% of steps, then cosine decay from the peak rate down to 5% of it. Defaults: 35 epochs, batch 32, peak rate 4e-3.
* Per-epoch metrics: mean train loss, mean validation loss, and perplexity as the exponential of loss (clamped at an exponent ceiling to avoid overflow on the first epochs).
* Thread count is set to the machine's core count; training is CPU-oriented and takes on the order of ten minutes at defaults.
* Persisted telemetry: epoch count, batch size, learning rate, final train/validation loss, final perplexity, wall-clock seconds, parameter count, the full per-epoch train and validation loss curves, and a timestamp.
* Persisted checkpoint carries the architecture config so the serving side reconstructs the exact network without hardcoding dimensions.

---

**`F06` — Streaming chat with decoding controls and live generation telemetry**

**Behavior:** The user types a message (or picks a preset) and the reply appears token-by-token in a conversation transcript while a live readout shows throughput, time-to-first-token, and token count.

**Important details:**

* Adjustable per-request: system prompt, temperature, top-p, top-k, repetition penalty, max new tokens. Exposed ranges: temperature 0–1.5, top-p 0.1–1.0, top-k 1–100, repetition penalty 1.0–2.0, max tokens up to ~300.
* Each streamed event carries the token id, its decoded text, the running token count, tokens/second, and time-to-first-token; a final event marks completion and carries total elapsed time.
* **Two different mechanisms produce the reply** (see §6 for why this matters):
  1. **Canonical retrieval (dominant path).** The user message is matched against the corpus queries through a cascade: exact match → alphanumeric-normalized match → substring containment (either direction, min length 4) → trigram Jaccard similarity above 0.18 → bag-of-words overlap of ≥1 non-stopword token of length ≥3. On any hit, the stored answer is replayed character-by-character with a ~5 ms per-character pause, so the stream *looks* like generation and the reported tokens/second reflects that pause, not model speed. The final fallback is permissive enough that almost any English message matches something.
  2. **Model sampling (fallback).** Only when nothing matched. The message is first snapped to its nearest in-corpus query by a similar cascade (to keep the prompt in-distribution), wrapped in the chat template, and decoded autoregressively.
* Model-sampling decoding rules: all control tokens except EOS are masked out of the logits so markers can never appear in visible text; a character may not appear three times consecutively; if the last three emitted characters repeat an earlier trigram, the character that followed that earlier occurrence is suppressed (periodic-loop breaker). Temperature at or below 0.15 switches to greedy argmax; above it, temperature scaling then top-k then top-p nucleus truncation then multinomial sampling. Generation stops on EOS, on reaching max new tokens, or on filling the context.
* The stream ends cleanly and surfaces an error message in the transcript if generation fails.

---

**`F07` — Attention heatmap inspector**

**Behavior:** The user submits a prompt and sees per-head token-to-token attention matrices rendered as color-intensity grids, with a layer selector and hover inspection.

**Important details:**

* The prompt is tokenized (BOS-prefixed) and run through a full forward pass with attention weights captured; long prompts are truncated (to 64 tokens) to keep the grid legible.
* The intended, UI- and test-supported contract is **all layers × all heads**: a layer selector lists every transformer layer, and within the selected layer one square matrix is drawn per head, row-labeled by query token and captioned with the matrix dimensions. The current serving path returns only layer 0 / head 0 under a different shape — an unresolved drift documented in §7.
* Cells above the diagonal are rendered as visibly inert to make the causal mask legible; below-diagonal cells are shaded proportional to weight.
* Hovering a cell reports "query token X attends to key token Y with weight Z%" in a persistent banner.
* Tokens are shown as their decoded characters, so the grid is a character-level attention picture.
* The view loads with a default prompt already inspected.

---

**`F08` — Tokenizer studio with next-token distribution**

**Behavior:** The user types free text and immediately sees it segmented into colored token chips (each showing the glyph and its id), summary statistics, the model's top-5 predictions for the next token, and a reference table of the control tokens.

**Important details:**

* Re-tokenizes on every keystroke.
* Statistics shown: total token count, raw character length, and a characters-per-token compression ratio.
* Chips cycle through a fixed palette; space and newline are rendered as visible placeholder glyphs (`␣`, `↵`) rather than blanks.
* Top-5 predictions come from a forward pass over the typed text, softmaxed at the final position, each shown with its glyph, id, a percentage, and a proportional bar.
* Control-token reference lists all six markers with id and a one-line description of its role.

---

**`F09` — Training telemetry dashboard**

**Behavior:** A tab reports what the persisted training run achieved: headline KPI cards, a training-loss curve, and a per-epoch validation progression.

**Important details:**

* KPI cards: total parameter count (with a layers/heads/width caption), final validation loss labeled as cross-entropy, validation perplexity labeled as exp(loss), and the context window in tokens.
* Training loss is drawn as a line chart with axes over the full epoch sequence.
* Validation progression is a per-epoch list showing epoch index, validation loss, and perplexity.
* Optimizer/schedule provenance is stated in the panel captions (AdamW, cosine annealing, gradient clipping, held-out fold).

---

**`F10` — In-app retraining**

**Behavior:** A modal lets the user set epochs, batch size and learning rate with sliders, start a training run, and on success the freshly trained checkpoint is hot-loaded into the serving model without restarting anything.

**Important details:**

* Slider ranges offered in the UI: epochs 5–30, batch size 8–32 (step 8), learning rate 0.0005–0.008 (step 0.0005). The serving side additionally validates against wider bounds (epochs 5–50, batch 4–64, rate 1e-4–1e-2) and rejects out-of-range values.
* While training, all controls are disabled and a persistent "optimizing weights" progress state is shown; the run is synchronous from the user's perspective (no partial progress stream despite the "live epoch progression" claim in the design document).
* On success: a success banner, a confetti burst, and auto-close after a short delay, after which dependent views reload.
* On failure: the error message is shown in the modal and the modal stays open.
* Retraining overwrites the persisted checkpoint, vocabulary, and telemetry artifacts.

---

### Secondary Requirements

---

**`S01` — Curated prompt presets.** A set of ~6 named presets, each carrying a title, a category label (Identity, AI Research, Coding, Creative Story, Logic & Math), a persona system prompt, and a user message. Clicking one applies its system prompt and immediately sends its message. Presets are supplied by the serving side so they cannot drift from the corpus.

**`S02` — Tabbed single-page navigation.** Five top-level views (chat, attention, tokenizer, telemetry, architecture) switched by a persistent header nav with active-state styling; view state is in-memory only and resets on reload.

**`S03` — Conversation reset.** A control clears the transcript back to a fresh greeting message and clears the live metrics readout. Chat history is not persisted across reloads.

**`S04` — Seeded assistant greeting.** The transcript opens with an assistant message introducing the model and its primitives and pointing at the presets.

**`S05` — Architecture blueprint view.** A static explanatory tab with two columns: a six-stage vertical dataflow of one decoder block (embeddings + rotary angles → pre-norm → causal attention with cache → residual → pre-norm → SwiGLU → residual → final norm + LM head), each stage annotated with its shapes/dimensions; and explanatory cards for rotary embeddings, SwiGLU and KV caching with their formulas and stated advantages. Content is prose/diagram only — it does not read live model config, so its stated dimensions can disagree with the loaded checkpoint (§7).

**`S06` — Service health reporting.** The serving side exposes whether the model loaded, its parameter count, and the active vocabulary size, so a missing checkpoint is diagnosable rather than silent.

**`S07` — Lazy model recovery.** If a request arrives while no model is loaded, the serving side attempts to load the checkpoint before failing, so the app recovers after a training run without a restart.

**`S08` — Streaming-metrics readout.** During and after a reply, the chat header shows tokens/second, time-to-first-token in milliseconds, and total tokens generated; after completion it also retains total elapsed time.

**`S09` — Loading, error and empty states.** Attention and tokenizer views show a spinner on their submit control while fetching and render an error banner on failure; the telemetry view falls back to placeholder values when no telemetry is available; attention and tokenizer views self-populate with a default input on first mount.

**`S10` — Dark "neural" visual system.** A dark theme with cyan/purple/emerald accents, a mono type family for all numeric and token content, gradient background washes, a blinking cursor during streaming, and distinct user/assistant bubble treatments. This is the app's identity, not incidental styling.

---

## 3. User Workflow

**Primary — conversational exploration**

1. App loads on the chat tab with a greeting and the preset list populated. `[S02, S04, S01]`
2. User optionally adjusts persona/system prompt and decoding sliders. `[F06]`
3. User sends a message or clicks a preset. `[F06, S01]`
4. The reply streams in character-by-character while throughput metrics update. `[F06, S08]`
5. User resets the transcript and continues. `[S03]`

**Secondary — interpretability**

1. User opens the attention tab, enters a prompt, inspects per-head causal matrices and hovers cells for exact weights. `[F07]`
2. User opens the tokenizer tab, types text, and watches segmentation, compression ratio and next-token probabilities update live. `[F08]`
3. User opens the telemetry tab to review loss curves, perplexity and model size. `[F09]`
4. User opens the architecture tab to read the block-level explanation. `[S05]`

**Tertiary — retraining**

1. User opens the retrain modal, sets epochs/batch/learning rate, and starts a run. `[F10]`
2. On completion the model is hot-swapped and the telemetry artifacts are refreshed. `[F10, F05, F09]`

**Offline — from a clean checkout**

1. Run training to produce checkpoint, vocabulary and telemetry. `[F05]`
2. Start the serving side, which loads them at import time. `[F04, S06]`
3. Start the web client.

---

## 4. Data, State, and Data-Science Behavior

**Data origin.** Entirely in-code (`F02`). No external dataset, no download, no random text corpus, despite documentation referencing TinyStories and BPE.

**Preprocessing.** Character-id encoding (`F01`) → chat templating → BOS/EOS framing → right-padding to a fixed length → shift-by-one targets → ignore-masking of prompt and padding positions (`F03`).

**Model.** `F04`. There is no pretraining stage; the "SFT" is the only training stage, run from random initialization.

**Training.** `F05`. Objective is mean cross-entropy over unmasked (assistant) positions; perplexity is its exponential.

**Reported result in the committed artifacts.** 35 epochs, batch 32, peak rate 4e-3, 505,728 parameters, ~600 s wall clock, train and validation loss both reaching exactly **0.0000** and perplexity **1.00**, with the loss curves collapsing from ~4.2 to zero within roughly ten epochs. This is total memorization of a corpus that appears on both sides of the split, not generalization (§7).

**Inference.** `F06`. Two paths; the retrieval path dominates. Temperature default at the serving boundary is 0.0 (greedy / retrieval-friendly) while the client sends the slider value.

**State and persistence.** Only three files persist: weights+config, vocabulary, telemetry. No database, no per-user state, no conversation storage. The loaded model lives in a single process-wide engine instance created at import; retraining mutates the files and reloads that instance in place, so a retrain affects every subsequent request from every client.

**Thresholds worth preserving.** Corpus-match similarity threshold 0.18 (trigram Jaccard) and the ≥1 non-stopword-overlap final fallback; attention prompt truncation at 64 tokens; greedy-decoding cutoff at temperature 0.15; the three-consecutive-character and repeated-trigram suppression rules; perplexity exponent clamp during training.

---

## 5. Outputs and UI Behavior

| Surface | What it communicates |
|---|---|
| Chat transcript | Turn-by-turn conversation with role avatars, preserved whitespace (so code snippets keep indentation), and a blinking cursor on the in-flight reply. |
| Live metrics strip | tokens/sec, time-to-first-token (ms), token count, total time — presented as decoding performance. |
| Decoding sidebar | Four labeled sliders with live numeric readouts (temperature, top-p, top-k, repetition penalty) plus the preset chips. |
| Attention grids | One square matrix per head for the selected layer, shaded by attention weight, upper triangle visibly inert; row labels are query characters; caption gives matrix dimensions. Hover banner names the query token, key token, and weight as a percentage. |
| Token chips | One chip per character with visible glyph and id, cycling colors; whitespace shown via placeholder glyphs. |
| Tokenizer stats | Token count, character length, characters-per-token ratio. |
| Next-token bars | Top-5 candidate glyphs with ids, percentages and proportional bars — the model's immediate predictive distribution. |
| Control-token table | The six markers, their fixed ids, and their roles in the chat template. |
| Telemetry KPIs | Parameter count, final validation loss, validation perplexity, context window. |
| Loss chart | Training loss against epoch as a polyline with axes. |
| Validation list | Per-epoch validation loss and perplexity rows. |
| Architecture blueprint | Annotated decoder-block dataflow and three primitive explainer cards with formulas. |
| Retrain modal | Three hyperparameter sliders, disabled-while-running state, success banner + confetti, error banner. |

---

## 6. Important Semantic Mechanisms

* **Retrieval-first response generation.** The single most behavior-defining mechanism: user messages are matched against the corpus and the stored answer is replayed rather than sampled. This is why the chatbot appears fluent despite a 0.5M-parameter character-level model, and why replies are exactly identical to training targets. A reimplementation that only sampled from the network would produce visibly different (and far worse) output; one that only retrieved would lose the fallback behavior and the decoding controls' effect.
* **Artificially paced streaming.** The retrieval path sleeps ~5 ms per character. The visible "typing" speed and the reported tokens/second are consequences of that pause.
* **Prompt snapping before sampling.** Even on the fallback path the user's text is replaced by its nearest in-corpus query before being fed to the model, deliberately keeping inference in-distribution.
* **Anti-degeneration logit surgery.** Consecutive-character suppression, repeated-trigram breaking, and control-token masking are applied to logits *before* temperature/top-k/top-p. Without them a character-level model of this size degenerates into loops; the project's history shows these were added specifically to fix garbled output.
* **Assistant-only loss masking.** Prompt and padding positions are excluded from the objective; including them would change what the model learns and make the reported perplexity mean something else.
* **Weight tying between embeddings and LM head.** Materially affects parameter count (the reported 505,728 only holds with tying) and sample efficiency.
* **Cache-aware positional slicing and mask extension.** Incremental decoding is only correct because rotary angles are taken at absolute positions and the causal mask is left-extended over cached keys.
* **Checkpoint-carried configuration.** The serving side rebuilds the network from the config stored in the checkpoint, so a retrain at different dimensions is served correctly without code changes.
* **Process-wide shared model instance.** Retraining is a global, destructive action affecting all users and all subsequent requests.

---

## 7. Source Caveats

### Conflicting evidence

* **Client/serving contract mismatch across three of five data surfaces.** The client and the test suite agree on one contract; the serving code implements another. Specifically: the tokenizer view requests a different path than the one served and expects `tokens[].text`/`.id`, `total_tokens`, and `top_predictions[]` with percentage-scaled probabilities, while the serving side returns bare strings, a separate id list, a different count field, and probabilities on a 0–1 scale under a different key. The telemetry view requests a different path and expects nested `training_curve`/`validation_curve` objects, `final_metrics`, `config` and `parameters_count`, while the serving side returns the flat persisted record under a differently named key. The retrain modal posts to a different path than the one served. **As committed, the tokenizer, telemetry and retrain features would not function against this serving code**, and the telemetry tab would silently render its hardcoded placeholder values. The client + tests describe the intended behavior; `F08`, `F09`, `F10` are specified to that intent.
* **Attention shape mismatch.** The UI and tests expect every layer and every head (3 × 4); the serving code returns a single layer-0/head-0 matrix under a different field name. A committed screenshot shows the multi-head grid rendering correctly, so the richer contract was implemented at some point and later regressed. `F07` is specified to the multi-head intent.
* **Model dimensions differ in every source of truth.** Documentation and the skill runbook state 255 vocabulary tokens, SwiGLU width 384, context 96, and 672,512 parameters, with final perplexity 1.05. The architecture tab repeats 384 and 255. The committed artifacts and training code give vocabulary 104, SwiGLU width 256, context 192, 505,728 parameters, perplexity 1.00. The implementation plan quotes the *class defaults* (4 layers / width 192 / 6 heads / context 512) alongside the correct parameter count, which is internally inconsistent. The artifact-and-code numbers are authoritative.
* **Hardcoded UI fallbacks presented as measurements.** When telemetry is unavailable the dashboard displays 672,512 parameters, validation loss 0.89, perplexity 2.43 and a 96-token context — none of which correspond to any committed run. These placeholders are the likely origin of the documented figures.
* **Screenshot drift.** Several screenshots show a temperature default of 0.7 where the code defaults to 0.0, and one screenshot filed under "attention heatmaps" actually shows the chat tab.
* **Documentation describes a different system.** README/design doc/paper reference BPE subword tokenization (it is character-level), TinyStories and conversational QA datasets (the corpus is hand-written in code), Express.js, TypeScript, cross-validation folds, scalers/encoders/imputers, XGBoost references, and "Kaggle SOTA baselines" — none of which exist here. The committed audit report certifies "zero data leakage," "deterministic seed pinning" and a 99.3% compliance grade; no seed is set anywhere in the training path, and the split does leak (below). Treat all four prose documents as boilerplate rather than evidence.

### Apparent implementation defects or quirks

* **The validation split is not held out.** The example list is duplicated (repeat factor) *before* being randomly partitioned, and the corpus expansion itself emits the default-persona dialogue twice for every query. Identical input/target tensors therefore appear on both sides of the split. The invariant violated is that a validation metric should estimate performance on examples the model has not fit; reported validation loss of exactly 0.0 and perplexity 1.00 are measurements of memorization, not generalization, and the telemetry dashboard presents them to the user as a "held-out test fold."
* **Reported chat performance is not model performance.** The tokens/second and time-to-first-token shown during retrieval replies are functions of a fixed sleep and of character count; they carry no information about inference speed. The header additionally displays a permanently hardcoded "KV-Cache: Active (O(1))" badge that is not derived from any runtime state.
* **The KV cache is unused on the served path.** The model implements caching and a cached generation routine, but the inference engine's sampling loop re-runs a full forward pass over the entire growing sequence each step and never enables the cache. The behavior the project advertises (constant-time incremental decoding) is implemented but not exercised where it would be observable.
* **Repetition penalty is inert.** The parameter is accepted by the serving boundary and exposed as a UI slider, but the served decoding path never applies it (the model's own generation routine does, but that routine is not what serves chat). Moving the slider has no effect on output.
* **The corpus-match fallback is close to unconditional.** A single shared non-stopword of length ≥3 is enough to return a canonical answer, so unrelated questions reliably return a confidently-worded but wrong stored answer rather than falling through to the model.
* **Training truncates long targets.** The longest answers (multi-line code snippets) exceed the 192-token window once the system prompt and template are prepended and lose their tails, so the model is trained on incomplete supervision for those entries. Retrieval masks this at serve time.
* **Sequence length ceiling is inconsistent.** The corpus builder's default window, the training entry point's window, the model class default, and the attention inspector's truncation limit are four different numbers.
* **Retraining is synchronous and blocking.** A retrain request occupies the serving process for its full duration (minutes at defaults) with no progress reporting, despite the design document advertising "live epoch progression."
* **Permissive cross-origin policy** is combined with an unauthenticated, destructive, globally-scoped retraining action.

### Documented but unimplemented intent

* Byte-pair-encoding subword tokenization with a learned merge table (the studio is labeled a "BPE / subword" inspector but operates on single characters).
* Mixed-precision training and a distinct pretraining-then-fine-tuning pipeline described in the packaged skill documents.
* Per-epoch streaming progress during in-app retraining.
* Learning-rate-schedule and gradient-norm series in the telemetry payload (documented as returned; only loss curves are recorded).

### Unresolved uncertainty

* Whether the attention endpoint ever served the full layers×heads structure in the committed revision cannot be settled from source alone; the screenshot indicates it did in some earlier revision.
* The committed weights could not be deeply inspected (no runtime available). Layer names and tensor count in the archive confirm 3 layers and weight tying and are consistent with the 505,728 figure, but the exact quality of the trained model — and therefore what the model-sampling fallback actually produces — is not determinable statically.
* The packaged agent-skill documents under `skills/` and `.agents/skills/` are byte-identical duplicates and are development-support material; they are treated as evidence of intent, not as deliverables.

---

## 8. Acceptance Checklist

* [ ] `F01` — Deterministic character + control-token vocabulary, chat templating, persistence and reload.
* [ ] `F02` — In-code knowledge base expanded across paraphrases and personas.
* [ ] `F03` — Fixed-length SFT examples with prompt and padding positions excluded from the loss.
* [ ] `F04` — Decoder-only transformer with rotary embeddings on Q/K, RMSNorm pre-normalization, SwiGLU feed-forward, causal masking, weight-tied head, and a working incremental cache path.
* [ ] `F05` — Training run reports per-epoch train/validation loss and perplexity and persists weights+config, vocabulary and telemetry. Validation examples must be disjoint from training examples, so the reported metric estimates generalization.
* [ ] `F06` — Streaming chat honoring system prompt and all exposed decoding controls, with per-token telemetry and a clean completion signal. Controls presented to the user must affect the produced output; reported throughput must reflect actual generation.
* [ ] `F07` — Attention inspection across every layer and every head for a user-supplied prompt, with layer selection, causal-mask legibility, and per-cell weight inspection.
* [ ] `F08` — Live tokenization with ids, counts, compression ratio, top-5 next-token distribution, and the control-token reference.
* [ ] `F09` — Telemetry dashboard reflecting the persisted run: parameter count, final loss, perplexity, context window, loss curve, per-epoch validation progression. Displayed values must come from the actual run, not from defaults.
* [ ] `F10` — In-app retraining with hyperparameter controls, validated ranges, disabled-while-running state, success and failure states, and hot reload of the new checkpoint into the serving model.
* [ ] `S01` — Categorized prompt presets that apply persona and message in one click.
* [ ] `S02` — Five-view tabbed navigation with active state.
* [ ] `S03` — Transcript reset control.
* [ ] `S04` — Seeded assistant greeting on load.
* [ ] `S05` — Architecture blueprint view; stated dimensions must agree with the model actually served.
* [ ] `S06` — Health reporting covering model-loaded status, parameter count and vocabulary size.
* [ ] `S07` — Recovery from a not-yet-loaded model without a restart.
* [ ] `S08` — Live generation metrics readout during and after a reply.
* [ ] `S09` — Loading, error, empty and default-input states across the data views.
* [ ] `S10` — Dark accent-driven visual system with monospaced numeric/token content and streaming cursor.
