import json
from pathlib import Path
from collections import Counter

eps = ["episode-001", "episode-002", "episode-005"]

total_calls = 0
total_in_tokens = 0
total_out_tokens = 0
total_lat = 0.0
resolved_after_count = 0
variants_count = 0

hooks = []
cliffs = []
clarities = []
payoffs = []
final_scores = []

all_ranked = []

for ep in eps:
    cand_p = Path(f"data/candidates/{ep}.json")
    judg_p = Path(f"data/judged/{ep}.json")
    
    cand_data = json.loads(cand_p.read_text(encoding="utf-8")) if cand_p.exists() else {}
    judg_data = json.loads(judg_p.read_text(encoding="utf-8")) if judg_p.exists() else {}
    
    cand_map = {c["candidate_id"]: c for c in cand_data.get("candidates", [])}
    
    survs = judg_data.get("judged", [])
    for j in survs:
        cid = j.get("candidate_id")
        orig = cand_map.get(cid, {})
        low_conf = orig.get("low_confidence", False)
        
        total_calls += 1
        total_in_tokens += j.get("llm_input_tokens", 0)
        total_out_tokens += j.get("llm_output_tokens", 0)
        total_lat += j.get("llm_latency_s", 0.0)
        
        h = j.get("hook", 0)
        c = j.get("cliff", 0)
        cl = j.get("clarity", 0)
        p = j.get("payoff", 0)
        fs = j.get("final_score", 0)
        ra = j.get("resolved_after", False)
        if ra:
            resolved_after_count += 1
        if j.get("is_variant", False):
            variants_count += 1
            
        hooks.append(h)
        cliffs.append(c)
        clarities.append(cl)
        payoffs.append(p)
        final_scores.append(fs)
        
        st = j.get("start_s", 0.0)
        en = j.get("end_s", 0.0)
        dur = j.get("duration_s", 0.0)
        s3a_sc = j.get("s3a_score", orig.get("score", 0.0))
        
        all_ranked.append({
            "ep": ep,
            "cid": cid,
            "st": st,
            "en": en,
            "dur": dur,
            "hook": h,
            "cliff": c,
            "clarity": cl,
            "payoff": p,
            "final_score": fs,
            "s3a_score": s3a_sc,
            "low_confidence": low_conf,
            "passed": j.get("passed_judge_gates", False)
        })

# Rank all candidates by final_score desc, tie-break by s3a_score desc
all_ranked.sort(key=lambda x: (x["final_score"], x["s3a_score"]), reverse=True)

print("--- TELEMETRY ---")
print(f"Total judge calls: {total_calls}")
print(f"Total tokens: in={total_in_tokens}, out={total_out_tokens}, total={total_in_tokens+total_out_tokens}")
print(f"Total latency: {total_lat:.2f}s (mean: {total_lat/max(1, total_calls):.2f}s/call)")
print(f"resolved_after count: {resolved_after_count}")
print(f"variants count: {variants_count}")

print("\n--- SCORE DISTRIBUTIONS ---")
print(f"hook: {dict(Counter(hooks))}")
print(f"cliff: {dict(Counter(cliffs))}")
print(f"clarity: {dict(Counter(clarities))}")
print(f"payoff: {dict(Counter(payoffs))}")
print(f"final_score: {dict(Counter(final_scores))}")

print("\n--- RANKED CANDIDATES ---")
print("Format: [start_s-end_s, duration, hook, cliff, clarity, payoff, final_score, low_confidence]")
for r in all_ranked:
    print(f"[{r['st']:.1f}-{r['en']:.1f}s, {r['dur']:.1f}s, {r['hook']}, {r['cliff']}, {r['clarity']}, {r['payoff']}, {r['final_score']}, {r['low_confidence']}] (ep={r['ep']}, passed={r['passed']})")

