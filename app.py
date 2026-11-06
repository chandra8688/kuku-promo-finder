"""Streamlit Interactive Application for Promo Finder.

Local viewer for candidate inspection and evaluation. Run it on your own machine
only: it displays screenplay text from the gitignored outputs, so never deploy it
anywhere public.

Every number on screen is read from a file the pipeline wrote (eval_test.json,
eval_dev.json, labels.csv, run_manifest.json, config/scoring_v2.yaml). Nothing is
typed in here; when a file is missing the app says "not measured".

    pip install streamlit
    streamlit run app.py
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.labels import format_cell, load_labels, precision_at_k
from src.windows import scoring_provenance

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Kuku Promo Finder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .hook-badge {
        background-color: #dbeafe;
        color: #1e40af;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .cliff-badge {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    .beat-box {
        font-family: monospace;
        font-size: 0.82rem;
        padding: 6px 10px;
        margin-bottom: 4px;
        border-radius: 4px;
        background: #f1f5f9;
        border-left: 3px solid #94a3b8;
    }
    .beat-hook {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
    }
    .beat-cliff {
        background: #fffbeb;
        border-left: 3px solid #f59e0b;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def load_ranked_promos(base_dir: Path) -> List[Dict[str, Any]]:
    ranked_file = base_dir / "promos_ranked.json"
    if not ranked_file.exists():
        # Fallback to outputs
        ranked_file = Path("outputs") / "promos_ranked.json"
    if ranked_file.exists():
        with open(ranked_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("ranked_promos", data.get("promos", []))
    return []


def load_parsed_beats(base_dir: Path, episode_id: str) -> List[Dict[str, Any]]:
    for candidate_dir in [base_dir / "parsed", Path("data/parsed"), Path("outputs/parsed")]:
        p = candidate_dir / f"{episode_id}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("beats", [])
    return []


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:
        return None


def provenance_note(gold_csv: Path) -> str:
    try:
        with open(gold_csv, newline="", encoding="utf-8") as f:
            st_counts: Dict[str, int] = {}
            for r in csv.DictReader(f):
                k = (r.get("status") or "unknown").strip() or "unknown"
                st_counts[k] = st_counts.get(k, 0) + 1
    except Exception:
        return ""
    n = sum(st_counts.values())
    human = sum(v for k, v in st_counts.items() if k in ("human", "ai_proposed_human_reviewed"))
    parts = ", ".join(f"{v} {k}" for k, v in sorted(st_counts.items()))
    if human == n:
        return f"Reference provenance: {parts}; every window labelled or verified by a person."
    return f"Reference provenance: {parts}; {n - human} of {n} windows were never verified by a person."


def eval_table(path: Path, gold_csv: Path, in_sample: bool) -> None:
    data = load_json(path)
    if not data or not data.get("table"):
        st.info(f"Not measured: `{path.name}` not found. Produce it with "
                f"`python scripts/eval_v2.py --parsed data/parsed --annotated data/annotated "
                f"--gold {gold_csv.relative_to(ROOT).as_posix()} --out {path.name}`.")
        return
    st.table([{k: ("-" if v is None else str(v)) for k, v in row.items()} for row in data["table"]])
    note = provenance_note(gold_csv)
    if in_sample:
        note += " The weights were fitted on this set, so V2 numbers here are in-sample."
    st.caption(note + " Counts are pooled; CI columns are bootstrap over episodes of the per-episode rate, "
               "so a CI's point value can differ from the pooled fraction.")


def load_manifest(base_dir: Path) -> Dict[str, Any]:
    for mf in [base_dir / "run_manifest.json", Path("outputs/run_manifest.json")]:
        if mf.exists():
            with open(mf, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


# Sidebar Controls
st.sidebar.title("🎬 Promo Finder")
st.sidebar.caption("Microdrama Trailer & Promo Candidate System")

outputs_dir = Path("outputs_final")
if not outputs_dir.exists():
    outputs_dir = Path("outputs")

nav_choice = st.sidebar.radio(
    "Navigation",
    ["🏆 Top 15 Show Promos", "🔍 Candidate Explorer", "📊 System Evaluation", "⚙️ Pipeline Info"],
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**Brief Constraints:**
- Duration: **20s – 60s**
- Hook: **Lands in first 3.0s**
- Ending: **Unresolved beat (cliffhanger)**
- Commercial: **Show-level ranking**
"""
)

manifest = load_manifest(outputs_dir)
if manifest:
    st.sidebar.caption(
        f"Run ID: `{manifest.get('run_id', 'latest')}`\n\n"
        f"Calls: `{manifest.get('total_calls', 'n/a')}` | "
        f"Tokens: `{manifest.get('total_input_tokens', 'n/a')}`"
    )

# ---------------------------------------------------------------------------
# View 1: Top 15 Ranked Promos
# ---------------------------------------------------------------------------
if nav_choice == "🏆 Top 15 Show Promos":
    st.title("🏆 Top 15 Commercial Promo Ranking")
    st.write(
        "Ranked promo clips selected across the show, at most 2 per episode. "
        "Each candidate is scored for a hook in its first 3 seconds and an unresolved ending."
    )

    promos = load_ranked_promos(outputs_dir)
    if not promos:
        st.warning(f"No ranked promos found in `{outputs_dir}`. Run `python run.py --input data/raw` first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top Promos", f"{len(promos)}")
        c2.metric("Target Duration", "20 – 60s")
        res = precision_at_k(promos, load_labels(ROOT / "labels.csv"), k=15)
        short = f"{res['n_approved']}/{res['n_labelled']}" + ("" if res["complete"] else f" (partial, {res['n_labelled']} of {res['n_in_list']})")
        c3.metric("Precision@15 (human labels)", short if res["n_labelled"] else "not labelled")
        c4.metric("Avg Duration", f"{sum(p.get('duration_s', 0) for p in promos)/len(promos):.1f}s")

        st.divider()

        for idx, p in enumerate(promos, 1):
            ep = p.get("episode", "unknown")
            cid = p.get("candidate_id", f"promo_{idx}")
            st_s = p.get("start_s", 0.0)
            en_s = p.get("end_s", 0.0)
            dur_s = p.get("duration_s", 0.0)
            hook_text = p.get("hook_line", "")
            cliff_text = p.get("cliffhanger_line", "")
            scores = p.get("scores", {})

            with st.expander(
                f"#{idx:02d} | Episode {ep} [{st_s:.1f}s – {en_s:.1f}s ({dur_s:.1f}s)]",
                expanded=(idx <= 3),
            ):
                col_info, col_beats = st.columns([1, 2])

                with col_info:
                    st.markdown(f"**Candidate:** `{cid}`")
                    st.markdown(f"**Duration:** `{dur_s:.1f}s` (start: `{st_s:.1f}s`, end: `{en_s:.1f}s`)")
                    st.markdown(f"**Scores:** Hook: `{scores.get('hook', '-')}`/3 | Cliff: `{scores.get('cliff', '-')}`/3")
                    st.markdown(f"**Cold-Viewer Clarity:** `{scores.get('clarity', '-')}`/3")
                    if p.get("why"):
                        st.caption(f"**Judge's one-line why:** {p.get('why')}")

                with col_beats:
                    st.markdown("**Opening Hook Line:**")
                    st.info(f"🎣 {hook_text if hook_text else 'Visual / Action Cold Open'}")
                    st.markdown("**Closing Cliffhanger Line:**")
                    st.warning(f"⚡ {cliff_text if cliff_text else 'Visual / Musical Cliffhanger Cut'}")

# ---------------------------------------------------------------------------
# View 2: Candidate Explorer & Beat Inspector
# ---------------------------------------------------------------------------
elif nav_choice == "🔍 Candidate Explorer":
    st.title("🔍 Candidate Screenplay Beat Inspector")
    st.write("Inspect how the candidate window aligns with raw screenplay beats and timing markers.")

    promos = load_ranked_promos(outputs_dir)
    if not promos:
        st.warning(f"No ranked promos found in `{outputs_dir}`.")
    else:
        promo_options = {
            f"#{i+1:02d} - {p.get('episode')} ({p.get('start_s')}s - {p.get('end_s')}s)": p
            for i, p in enumerate(promos)
        }
        selected_label = st.selectbox("Select Candidate Promo:", list(promo_options.keys()))
        promo = promo_options[selected_label]

        ep_id = promo.get("episode")
        st_s = float(promo.get("start_s", 0.0))
        en_s = float(promo.get("end_s", 0.0))

        beats = load_parsed_beats(outputs_dir, ep_id)
        if not beats:
            st.error(f"Could not load parsed screenplay beats for `{ep_id}`.")
        else:
            in_window = [
                b for b in beats
                if b.get("end_s", 0) > st_s + 1e-4 and b.get("start_s", 0) < en_s - 1e-4
            ]

            st.write(f"Showing **{len(in_window)} beats** inside window `[{st_s:.2f}s – {en_s:.2f}s]`:")

            for b in sorted(in_window, key=lambda x: (x.get("start_s", 0), x.get("end_s", 0))):
                b_st = b.get("start_s", 0.0)
                b_en = b.get("end_s", 0.0)
                b_kind = b.get("kind", "")
                speaker = b.get("speaker_norm") or b_kind
                rel_t = max(0.0, b_st - st_s)
                text = (b.get("text") or "").strip()

                is_hook = rel_t < 3.0
                is_cliff = (en_s - b_en) < 3.0

                badge = ""
                box_cls = "beat-box"
                if is_hook:
                    badge = '<span class="hook-badge">HOOK [0-3s]</span> '
                    box_cls += " beat-hook"
                elif is_cliff:
                    badge = '<span class="cliff-badge">CLIFFHANGER</span> '
                    box_cls += " beat-cliff"

                st.markdown(
                    f'<div class="{box_cls}">'
                    f'<b>+{rel_t:04.1f}s</b> [{b_st:.1f}s - {b_en:.1f}s] '
                    f'{badge}<b>{html.escape(str(speaker))}</b>: {html.escape(text)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ---------------------------------------------------------------------------
# View 3: System Evaluation
# ---------------------------------------------------------------------------
elif nav_choice == "📊 System Evaluation":
    st.title("📊 System Evaluation")
    st.write("Read from the files `scripts/eval_v2.py` and `scripts/review.py` wrote. Nothing on this page is typed in.")

    st.subheader("1. Show-level precision@15")
    promos = load_ranked_promos(outputs_dir)
    res = precision_at_k(promos, load_labels(ROOT / "labels.csv"), k=15) if promos else None
    col1, col2 = st.columns(2)
    col1.metric("Precision@15 (human labels)", format_cell(res) if res and res["n_labelled"] else "not labelled")
    col2.metric("Baseline B1", "n/a", "one unranked clip per episode, cannot rank")
    st.caption("Labels come from a person reading each clip's beats in `scripts/review.py`, with model scores hidden.")

    st.divider()
    st.caption("Two independent annotation runs, same model and prompt: run A (used to fit the weights) and "
               "run B (the run that produced the shipped promos). Code and weights are identical; only the "
               "model's beat annotations differ.")
    st.subheader("2. Reserved test set: annotation run A")
    eval_table(ROOT / "eval_test.json", ROOT / "gold" / "gold_public.csv", in_sample=False)
    st.subheader("3. Reserved test set: annotation run B (shipped)")
    eval_table(ROOT / "eval_test_run_b.json", ROOT / "gold" / "gold_public.csv", in_sample=False)

    st.subheader("4. Development set: run A")
    eval_table(ROOT / "eval_dev.json", ROOT / "gold" / "gold_dev.csv", in_sample=True)
    st.subheader("5. Development set: run B")
    eval_table(ROOT / "eval_dev_run_b.json", ROOT / "gold" / "gold_dev.csv", in_sample=True)

    st.subheader("6. Annotation stability, run A vs run B")
    stab = load_json(ROOT / "annotation_stability.json")
    if not stab:
        st.info("Not measured: produce it with `python scripts/annotation_stability.py --parsed data/parsed "
                "--a data/annotated --b outputs_final/annotated --out annotation_stability.json`.")
    else:
        n = stab["episodes"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top-1 same clip (IoU >= 0.5)", f"{stab['top1_same_clip_iou50']}/{n}")
        c2.metric("Top-1 same window", f"{stab['top1_same_id']}/{n}")
        c3.metric("Comparable beats", f"{stab['comparable_beats']}")
        st.table([{"field": k, "exact agreement": f"{v * 100:.1f}%"} for k, v in stab["exact_agreement"].items()])

# ---------------------------------------------------------------------------
# View 4: Pipeline Info
# ---------------------------------------------------------------------------
elif nav_choice == "⚙️ Pipeline Info":
    st.title("⚙️ Pipeline Architecture & Integrity")

    prov = scoring_provenance()
    st.markdown(
        """
### End-to-End Execution Flow
```
Screenplay (.txt)
      │
      ▼
Stage 1: Deterministic parser (centisecond timings)
      │
      ▼
Stage 2: Beat-level annotation, one model call per episode (role: cheap)
      │
      ▼
Stage 3-4: Candidate engine, src/engine.py: full-span window scorer + structural gates
      │
      ▼
Stage 5: Judge (role: judge): writes the one-line why; does not affect ranking;
         removes a clip only when it and Stage 2 both say the tension resolves
      │
      ▼
Stage 6: Show-level ranker (S3 score, <= 2 per episode, low-confidence last)
      │
      ▼
Stage 7: Boundary refinement after ranking is frozen
```
"""
    )
    st.markdown(
        f"""
### Config provenance (computed live)
- `{prov["config_path"]}` sha256 `{prov["config_sha256"]}` (line-ending independent)
- Compare with `scoring_provenance.config_sha256` in any candidates file or `run_manifest.json`.

### Verify it yourself
```
python -m pytest tests/ -q
python scripts/leak_check.py
python scripts/check_submission.py
python run.py --input data/raw --out outputs_final
```
"""
    )
