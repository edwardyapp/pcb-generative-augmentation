"""One-line-per-check summary of the conditioning trend, for status.sh."""
import json
import os

TARGETS = (120, 160, 200, 240, 280, 320)

for path, label in (('results/cond_checks.jsonl', ''),
                    ('results/cond_checks_budget.jsonl', ' [budget generators]')):
    if not os.path.exists(path):
        continue
    rs = [json.loads(l) for l in open(path) if l.strip()]
    for r in sorted(rs, key=lambda r: (r.get('tag', ''), r['epoch'])):
        flag = 'COLLAPSED' if r['collapsed'] else 'ok'
        tag = r.get('tag', '')
        print(f"   ep {r['epoch']:>3} {tag:<12} consistency {100*r['consistency']:5.1f}%   "
              f"ex-collapse {100*r['consistency_ex']:5.1f}%   V {r['cramers_v']:.3f}   "
              f"{r['collapse_class']} {100*r['collapse_index']:.0f}%  [{flag}]{label}")

if os.path.exists('results/cond_checks.jsonl'):
    rs = [json.loads(l) for l in open('results/cond_checks.jsonl') if l.strip()]
    seen = {r['epoch'] for r in rs if r.get('tag') == 'condtight3'}
    todo = [e for e in TARGETS if e not in seen]
    print(f"   remaining: {todo if todo else 'none — gate can fire'}")
else:
    print('   (none yet)')
