import json
from pathlib import Path

# Collect 40 non-reserved episodes
reserved = {f"episode-{i:03d}" for i in (3, 8, 13, 18, 23, 28, 33, 38, 43, 48)}
active_eps = [f"episode-{i:03d}" for i in range(1, 51) if f"episode-{i:03d}" not in reserved]

# 1. Candidate survivors and fallbacks
survivors_per_ep = {}
fallback_episodes = {}
episodes_under_2_survivors = []

total_survivors = 0
total_fallbacks = 0

for ep in active_eps:
    cp = Path(f"data/candidates/{ep}.json")
    if not cp.exists():
        continue
    d = json.loads(cp.read_text(encoding="utf-8"))
    survs = [c for c in d.get("candidates", []) if c.get("passed_all_gates")]
    fbs = [c for c in survs if c.get("low_confidence")]
    n_surv = len(survs)
    n_fb = len(fbs)
    survivors_per_ep[ep] = n_surv
    if n_fb > 0:
        fallback_episodes[ep] = n_fb
    if n_surv < 2:
        episodes_under_2_survivors.append(ep)
    total_survivors += n_surv
    total_fallbacks += n_fb

# 2. Annotation calls, tokens, latency, cached vs non-cached, partial flags
ann_report_p = Path("annotate_report.json")
ann_report = json.loads(ann_report_p.read_text(encoding="utf-8")) if ann_report_p.exists() else {}
ep_reports = ann_report.get("episodes", [])
ep_map = {r["episode"]: r for r in ep_reports}

cached_calls = 0
non_cached_calls = 0
cached_in_tok = 0
cached_out_tok = 0
non_cached_in_tok = 0
non_cached_out_tok = 0
non_cached_lat = 0.0
partial_flags = 0

for ep in active_eps:
    r = ep_map.get(ep, {})
    is_cached = r.get("cached", False)
    c = r.get("calls", 0)
    tin = r.get("input_tokens", 0)
    tout = r.get("output_tokens", 0)
    lat = r.get("latency_s", 0.0)
    if r.get("partial", False):
        partial_flags += 1

    if is_cached:
        cached_calls += c
        cached_in_tok += tin
        cached_out_tok += tout
    else:
        non_cached_calls += c
        non_cached_in_tok += tin
        non_cached_out_tok += tout
        non_cached_lat += lat

print("=== PART B EMPIRICAL REPORT ===")
print(f"Total non-reserved episodes evaluated: {len(active_eps)}")
print(f"Total survivors across 40 episodes: {total_survivors} (mean: {total_survivors/len(active_eps):.2f}/ep)")
print(f"Episodes using fallback: {len(fallback_episodes)} episodes, {total_fallbacks} fallback candidates total")
print(f"Fallback episodes detail: {fallback_episodes}")
print(f"Episodes with fewer than 2 survivors: {len(episodes_under_2_survivors)} ({episodes_under_2_survivors})")
print("\n--- SURVIVORS PER EPISODE ---")
for ep, count in survivors_per_ep.items():
    fb_note = f" (fallbacks: {fallback_episodes.get(ep, 0)})" if ep in fallback_episodes else ""
    print(f"  {ep}: {count} survivors{fb_note}")

print("\n--- S2 ANNOTATION CALLS & TELEMETRY ---")
print(f"Total calls: {cached_calls + non_cached_calls} (Cached: {cached_calls}, Non-cached: {non_cached_calls})")
print(f"Total tokens: In={cached_in_tok + non_cached_in_tok}, Out={cached_out_tok + non_cached_out_tok}")
print(f"Cached tokens: In={cached_in_tok}, Out={cached_out_tok}")
print(f"Non-cached tokens: In={non_cached_in_tok}, Out={non_cached_out_tok}")
print(f"Non-cached latency: {non_cached_lat:.2f}s (mean: {non_cached_lat/max(1, non_cached_calls):.2f}s/call)")
print(f"Partial flags: {partial_flags}")

