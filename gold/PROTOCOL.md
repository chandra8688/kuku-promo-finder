# Gold Annotation Protocol

Specification and operational rules for creating reference promo clip annotations for microdrama screenplays.

---

## 1. Scope & Objectives

The goal is to annotate ground-truth promo clip windows that satisfy the four core creative and commercial requirements of microdrama marketing:
1. **Duration**: 20.0 to 60.0 seconds total continuous runtime.
2. **Hook (First 3.0s)**: High-salience narrative, emotional, or auditory trigger in the initial 3 seconds.
3. **Cliffhanger Ending**: Cuts abruptly on maximum unresolved dramatic tension or unanswered question.
4. **Stand-Alone Clarity**: Understandable to a cold viewer with no preceding episode knowledge.

Reference windows are used exclusively for objective benchmark evaluation and hyperparameter tuning.

---

## 2. Definitional Criteria

### 2.1 Hook Definition (Start to +3.0s)
A valid hook must capture attention within 3.0 seconds of clip start:
- **Eligible Hook Triggers**:
  - Direct confrontation, accusation, or ultimatum in dialogue.
  - High-intensity emotional reaction (grief, outrage, shock, betrayal).
  - Sudden physical action (slap, door burst, weapon reveal, object smash).
  - Sharp auditory stinger (SFX) coupled immediately with dramatic staging.
- **Ineligible Hook Triggers**:
  - Passive establishing camera shots (e.g. slow pans, empty room descriptions).
  - Low-stakes pleasantries or routine exposition without conflict.
  - Muffled or ambient background sounds without immediate foreground stakes.

### 2.2 Cliffhanger & Unresolved Ending Definition
A valid promo ending must terminate on a tension peak:
- **Unresolved Revelation**: A shocking fact or secret is exposed, and the clip cuts before the reaction or explanation.
- **Unanswered Question / Ultimatum**: A demand or question with severe consequences is posed, cutting before an answer.
- **Imminent Threat**: Physical, legal, or financial peril is initiated, cutting before resolution or escape.
- **Resolution Prohibition**: If the narrative immediately answers the question or diffuses the tension in subsequent lines, the clip boundary must terminate *before* that resolving beat.

### 2.3 Boundary Determination
- **Start Boundary ($t_{\text{start}}$)**:
  - Aligned to the start of the hook beat or opening spoken utterance.
  - Leading silence/dead-air must not exceed 1.0 second.
- **End Boundary ($t_{\text{end}}$)**:
  - Aligned immediately following the delivery of the cliffhanger line or sound cue.
  - Pad trailing spoken dialogue by 0.4s to 0.8s to avoid clipping the final syllable or breath.
  - Never cut mid-sentence or mid-word.
  - Trailing silence/dead air after the dramatic peak must not exceed 1.5 seconds.
- **Total Duration**:
  - Strictly $20.0\text{s} \le (t_{\text{end}} - t_{\text{start}}) \le 60.0\text{s}$.

### 2.4 Tie-Breaking Decision Rules
When multiple valid promo candidates exist within an episode:
1. **Hook Immediacy**: Prefer candidates with immediate verbal/action hooks within $\le 1.5\text{s}$ over hooks delayed toward $3.0\text{s}$.
2. **Tension Gradient**: Prefer clips that build steadily toward a sharper cliffhanger spike over flat dialogues.
3. **Stand-Alone Intelligibility**: Prefer clips whose conflict is self-evident (power struggle, betrayal, secret identity) over clips requiring detailed backstory.
4. **Diversity**: Annotate up to 2 distinct non-overlapping or distinct-arc promo candidates per episode when present.

---

## 3. Labeling Methodology & Workflow

To eliminate circularity and confirmation bias:
1. **No Pipeline Peek**: Screenplay scenes and beats are read and annotated from the parsed screenplay structure according to dramatic cues, without inspecting automated pipeline candidate outputs.
2. **Provenance Status**:
   - `human`: Fully human-identified and bounded from screenplay analysis.
   - `ai_proposed_human_reviewed`: Human-verified and adjusted reference window.
3. **Data Integrity & Confidentiality**:
   - Reference files record numerical identifiers and timestamps only (`gold_id`, `episode`, `seg_order`, `start_s`, `end_s`, `status`).
   - Dialogue snippets, character names, and plot summaries must never be entered into reference CSVs or version control.

---

## 4. Quality Control & Agreement Metric

- **Double-Pass Protocol**: At least 5 episodes are annotated independently in two distinct passes (or by two reviewers).
- **Agreement Metric**: Window-level Intersection-over-Union ($\text{IoU}$):
  $$\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
- A pair is considered in agreement if $\text{IoU} \ge 0.50$.
- Self-agreement IoU must be computed and published in the evaluation documentation.

---

## 5. Measured Double-Pass Agreement Results

Five dev episodes (`episode-002`, `episode-007`, `episode-010`, `episode-018`, `episode-046`) were annotated across two independent passes:

| Episode | Window ID | Pass 1 Window [s] | Pass 2 Window [s] | IoU | Agreement ($\ge 0.5$) |
|---|---|:---:|:---:|:---:|:---:|
| `episode-002` | `DEV_E02_C01` | [0.00, 37.20] | [0.00, 37.10] | 0.9973 | PASS |
| `episode-002` | `DEV_E02_C02` | [63.50, 107.40] | [63.50, 107.40] | 1.0000 | PASS |
| `episode-007` | `DEV_E07_C01` | [65.60, 107.80] | [65.20, 107.80] | 0.9906 | PASS |
| `episode-007` | `DEV_E07_C02` | [48.80, 91.40] | [48.80, 91.50] | 0.9977 | PASS |
| `episode-010` | `DEV_E10_C01` | [2.10, 57.60] | [2.10, 57.60] | 1.0000 | PASS |
| `episode-010` | `DEV_E10_C02` | [38.10, 85.90] | [38.10, 86.00] | 0.9979 | PASS |
| `episode-018` | `DEV_E18_C01` | [19.50, 62.50] | [20.80, 62.50] | 0.9698 | PASS |
| `episode-018` | `DEV_E18_C02` | [68.00, 115.50] | [68.00, 115.50] | 1.0000 | PASS |
| `episode-018` | `DEV_E18_C03` | [145.80, 186.50] | [145.80, 186.50] | 1.0000 | PASS |
| `episode-046` | `DEV_E46_C01` | [7.80, 58.00] | [11.50, 58.00] | 0.9263 | PASS |
| `episode-046` | `DEV_E46_C02` | [78.10, 123.60] | [78.10, 123.60] | 1.0000 | PASS |
| `episode-046` | `DEV_E46_C03` | [96.80, 143.80] | [96.80, 143.80] | 1.0000 | PASS |

- **Total Windows Evaluated**: 12
- **Windows in Agreement ($\text{IoU} \ge 0.50$)**: 12 / 12 (100.0%)
- **Mean Inter-Pass IoU**: **0.9900** (99.0%)

---

## 6. Dataset Split Integrity

- **Reserved Test Set (`gold/gold_public.csv`)**: Episodes 3, 13, 23, 33, 43 (7 windows). **Frozen and untouched.**
- **Development Set (`gold/gold_dev.csv`)**: Episodes 2, 5, 6, 7, 8, 9, 10, 17, 18, 20, 39, 46 (25 windows across 12 episodes).
- **Set Disjointness**: $\text{Dev} \cap \text{Test} = \emptyset$. Zero data leakage.

