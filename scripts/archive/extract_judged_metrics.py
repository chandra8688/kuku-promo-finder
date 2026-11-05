import json
from pathlib import Path

dev_eps = ['episode-001', 'episode-002', 'episode-005']
all_judged = []

total_in_tokens = 0
total_out_tokens = 0
total_lat = 0.0
total_calls = 0
resolved_after_count = 0
variants_generated = 0

score_dists = {
    'hook': {0: 0, 1: 0, 2: 0, 3: 0},
    'cliff': {0: 0, 1: 0, 2: 0, 3: 0},
    'clarity': {0: 0, 1: 0, 2: 0, 3: 0},
    'payoff': {0: 0, 1: 0, 2: 0, 3: 0},
}

for ep in dev_eps:
    p = Path(f'data/judged/{ep}.json')
    if not p.exists():
        continue
    data = json.load(open(p, 'r', encoding='utf-8'))
    judged_list = data.get('judged', [])
    for c in judged_list:
        all_judged.append(c)
        total_calls += 1
        total_in_tokens += c.get('llm_input_tokens', 0)
        total_out_tokens += c.get('llm_output_tokens', 0)
        total_lat += c.get('llm_latency_s', 0.0)
        if c.get('resolved_after'):
            resolved_after_count += 1
        if c.get('is_variant'):
            variants_generated += 1
        for field in ['hook', 'cliff', 'clarity', 'payoff']:
            val = c.get(field)
            if val is not None and val in score_dists[field]:
                score_dists[field][val] += 1

print('--- JUDGED SUMMARY ---')
print(f'Total calls: {total_calls}')
print(f'Total input tokens: {total_in_tokens}')
print(f'Total output tokens: {total_out_tokens}')
print(f'Total latency: {round(total_lat, 2)}s (mean {round(total_lat/max(1, total_calls), 2)}s/call)')
print(f'Total resolved_after: {resolved_after_count}')
print(f'Total variants generated: {variants_generated}')
print('Score distributions:')
for f, dist in score_dists.items():
    print(f'  {f}: {dist}')

print('\n--- FINAL RANKED CANDIDATES ---')
for ep in dev_eps:
    p = Path(f'data/judged/{ep}.json')
    if not p.exists():
        print(f'{ep}: 0 survivors.')
        continue
    data = json.load(open(p, 'r', encoding='utf-8'))
    judged_list = data.get('judged', [])
    # Sort descending by final_score, tie-break by s3a_score
    judged_list.sort(key=lambda x: (x.get('final_score', 0), x.get('s3a_score', 0.0)), reverse=True)
    print(f'{ep} ({len(judged_list)} candidate(s)):')
    for c in judged_list:
        se = f"{c['start_s']:.1f}-{c['end_s']:.1f}"
        dur = c.get('duration_s', round(c['end_s'] - c['start_s'], 1))
        h = c.get('hook')
        q = c.get('cliff')
        clarity = c.get('clarity')
        payoff = c.get('payoff')
        final_score = c.get('final_score')
        v_of = c.get('parent_id') or "none"
        print(f"  [{se}, {dur:.1f}s, {h}, {q}, {clarity}, {payoff}, {final_score}, {v_of}]")
