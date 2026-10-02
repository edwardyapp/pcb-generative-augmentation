"""THE GATE. Decides whether the budget-restricted generators get trained at all.

Written and committed BEFORE the epoch-320 conditioning check exists, so the bar cannot be
moved to fit the result.

Rationale for the criteria: at epoch 80 the naive "consistency" read 22.4% -- superficially
above the 16.7% chance line -- but that number was manufactured entirely by mode collapse
(61.5% of ALL samples were classified `spur`, whatever class was requested). Strip the
collapse row and consistency fell to 13.1%, BELOW chance. So a pooled number alone can never
be the gate. All three of the following must hold:

  (1) collapse_index <= 0.40   no single class may absorb >40% of all predictions
                               (uniform = 0.167; at epoch 80 it was 0.615)
  (2) consistency_ex >= 0.30   consistency with the collapse class's ROW removed must be
                               clearly above the 16.7% chance line, not marginally
                               (n=60 in that submatrix => SE ~ 6%, so 0.30 is ~2.2 SE up)
  (3) cramers_v     >= 0.30    association between conditioned and predicted class. This is
                               the collapse-proof statistic: unlike consistency it CANNOT be
                               inflated by the model dumping every sample in one class.
                               (at epoch 80: 0.216)

PASS  -> train the budget-restricted generators (10/25/50%).
FAIL  -> stop. If class conditioning does not work with the FULL train pool and a 320-epoch
         budget, it will not work on 215 crops. We ship the negative result.

Exit code 0 = PASS, 1 = FAIL, 2 = the epoch-320 check has not run yet.
"""
import sys
import json

CHECKS = 'results/cond_checks.jsonl'
GATE_EPOCH = 320
MAX_COLLAPSE = 0.40
MIN_CONSISTENCY_EX = 0.30
MIN_CRAMERS_V = 0.30


def main():
    try:
        rs = [json.loads(l) for l in open(CHECKS) if l.strip()]
    except FileNotFoundError:
        print(f'GATE: {CHECKS} does not exist yet'); return 2

    r = next((x for x in rs if x['epoch'] == GATE_EPOCH), None)
    if r is None:
        have = sorted(x['epoch'] for x in rs)
        print(f'GATE: no epoch-{GATE_EPOCH} check yet (have: {have})'); return 2

    tests = [
        ('collapse_index  <= 0.40', r['collapse_index'], r['collapse_index'] <= MAX_COLLAPSE),
        ('consistency_ex  >= 0.30', r['consistency_ex'], r['consistency_ex'] >= MIN_CONSISTENCY_EX),
        ("cramer's V      >= 0.30", r['cramers_v'], r['cramers_v'] >= MIN_CRAMERS_V),
    ]
    print(f'=== GATE @ epoch {GATE_EPOCH} (criteria fixed in advance) ===')
    print(f'  consistency (pooled, for reference only): {100*r["consistency"]:.1f}%')
    print(f'  collapse class: {r["collapse_class"]}')
    for name, val, ok in tests:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}   actual = {val:.3f}')

    passed = all(ok for _, _, ok in tests)
    print(f'\nGATE: {"PASS -> train budget-restricted generators" if passed else "FAIL -> stop; ship the negative result"}')
    if not passed:
        print('  (conditioning does not work with the FULL train pool and 320 epochs;')
        print('   it will not work when the generator is restricted to 215 crops.)')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
