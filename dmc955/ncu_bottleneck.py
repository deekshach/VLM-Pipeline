#!/usr/bin/env python3
"""
Name the binding hardware unit for every launch shape in an ncu survey.

WHY THIS EXISTS (read before trusting any earlier bottleneck claim)

The 2026-09-02 survey reported SM% / DRAM% / "L2%" / occ% / waves and concluded
that prefill's two largest GEMMs were bound by nothing ("45-48% SM and 32-37%
DRAM -- neither limit. Least-explained result in the survey.").

That conclusion was an artifact of the columns chosen, not of the hardware:

  1. The "L2%" column was **L2 Hit Rate**, not L2 throughput. Hit rate says where
     the traffic landed; it says nothing about whether that unit is saturated.
     `--set full` had captured "L2 Cache Throughput" all along -- it was never
     read. For prefill's ampere_64x128 it is 68.8% against DRAM's 32.4%.
  2. SOL "Memory Throughput" is max(DRAM, L2, L1), i.e. the dominant memory unit.
     Reading only DRAM% therefore reports ~0 for any kernel whose working set is
     served out of L2 -- which is exactly the high-hit-rate GEMMs.
  3. `Compute (SM) Throughput` aggregates every pipe. For a tensor-core GEMM the
     honest number is the tensor pipe utilisation, reported by ncu's
     HighPipeUtilization rule, not SM%.

So this tool reports, per launch: all four SOL throughputs, which unit the SOL
memory figure resolves to, the occupancy ceiling and *why* it is capped, and the
issue rate -- then names a verdict.

BUGS IN ncu_report.py THIS TOOL AVOIDS (both still present there)

  * Its glob is `ncu_*.csv`, but survey files are `<phase>__<kernel>__<share>.csv`.
    It matches nothing and prints 0% coverage **silently**. Run it and see.
  * Its WANT table matches bare metric names, but "Memory Throughput" appears
    twice -- SOL (%) and Memory Workload Analysis (Gbyte/s). Last row wins, so
    its `mem%` field actually holds GB/s. Key on (section, metric), as here.

Usage:
    python3 ncu_bottleneck.py [--dir ncu_2026-09-02_7B-BF16] [--rules]
"""
import csv, re, os, glob, sys, argparse, collections

SOL  = 'GPU Speed Of Light Throughput'
MEM  = 'Memory Workload Analysis'
COMP = 'Compute Workload Analysis'
SCH  = 'Scheduler Statistics'
LAU  = 'Launch Statistics'
OCC  = 'Occupancy'

# (section, metric) -> short key. Keyed on the PAIR: metric names are not unique.
WANT = {
    (SOL,  'Duration'):                      'dur',
    (SOL,  'Compute (SM) Throughput'):       'sm',
    (SOL,  'Memory Throughput'):             'mem',      # = max(dram, l2, l1)
    (SOL,  'DRAM Throughput'):               'dram',
    (SOL,  'L1/TEX Cache Throughput'):       'l1',
    (SOL,  'L2 Cache Throughput'):           'l2',
    (MEM,  'L2 Hit Rate'):                   'l2hit',
    (MEM,  'Mem Pipes Busy'):                'pipes',
    (COMP, 'Issue Slots Busy'):              'issue',
    (SCH,  'Eligible Warps Per Scheduler'):  'elig',
    (OCC,  'Achieved Occupancy'):            'occ',
    (OCC,  'Theoretical Occupancy'):         'occth',
    (OCC,  'Theoretical Active Warps per SM'): 'thwarps',
    (LAU,  'Waves Per SM'):                  'waves',
    (LAU,  'Registers Per Thread'):          'regs',
}


def load(path):
    launches = collections.OrderedDict()
    for r in csv.DictReader(l for l in open(path) if not l.startswith('==')):
        kid = r.get('ID')
        if not kid:
            continue
        e = launches.setdefault(kid, {'rules': []})
        if 'grid' not in e:
            n = [int(x) for x in re.findall(r'\d+', r.get('Grid Size', ''))]
            e['grid'] = n[0] * n[1] * n[2] if len(n) >= 3 else None
            e['name'] = r.get('Kernel Name', '')
        key = WANT.get((r.get('Section Name'), r.get('Metric Name')))
        if key:
            try:
                v = float(str(r.get('Metric Value')).replace(',', ''))
            except (TypeError, ValueError):
                v = None
            if v is not None:
                u = (r.get('Metric Unit') or '').lower()
                if key == 'dur':
                    v *= 1e3 if u.startswith('m') else (1e-3 if u.startswith('n') else 1)
                e[key] = v
        rn, rd = r.get('Rule Name'), r.get('Rule Description')
        if rn and rd and not any(a == rn and b == rd for a, b, _ in e['rules']):
            e['rules'].append((rn, rd, r.get('Estimated Speedup', '')))
    return launches


def pipe_util(e):
    """Tensor/ALU pipe utilisation, which ncu states only in prose."""
    for rn, rd, _ in e['rules']:
        if rn == 'HighPipeUtilization':
            m = re.search(r'([\d.]+) is the highest|highest-utilized pipeline \(([\d.]+)%\)', rd)
            if m:
                return float(m.group(1) or m.group(2)), (rd.split(' is the highest')[0].strip() or '?')
    return None, None


def verdict(e):
    """Name the binding unit. Thresholds stated, not implied."""
    units = {'DRAM': e.get('dram', 0), 'L2': e.get('l2', 0),
             'L1': e.get('l1', 0), 'SM': e.get('sm', 0)}
    unit = max(units, key=units.get)
    top = units[unit]
    if e.get('waves', 9e9) < 1.0:
        return f'UNDERFILLED ({e["waves"]:.2f} waves) — {unit} {top:.0f}%'
    if top >= 80:
        return f'SATURATED on {unit} ({top:.0f}%)'
    if top >= 60:
        return f'{unit}-leaning ({top:.0f}%)'
    return f'LATENCY-BOUND (nothing >60%; max {unit} {top:.0f}%)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ncu_2026-09-02_7B-BF16'))
    ap.add_argument('--rules', action='store_true', help='print ncu rule prose')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, '*.csv')))
    if not files:
        sys.exit(f'no CSVs in {a.dir}')

    order = {'vit': 0, 'prefill': 1, 'prefill_attn': 1, 'decode': 2}
    files.sort(key=lambda f: (order.get(os.path.basename(f).split('__')[0], 9),
                              os.path.basename(f)))

    hdr = (f"{'kernel':<34}{'grid':>8}{'us':>9}{'SM%':>6}{'tensor':>7}"
           f"{'DRAM%':>7}{'L2%':>6}{'L1%':>6}{'L2hit':>6}"
           f"{'occ/th':>9}{'w/sched':>8}{'elig':>6}{'issue%':>7}{'waves':>7}  verdict")
    cur = None
    for f in files:
        phase = os.path.basename(f).split('__')[0]
        if phase != cur:
            print(f'\n{"="*len(hdr)}\n== {phase.upper()}\n{"="*len(hdr)}')
            print(hdr)
            cur = phase
        print(f'\n-- {os.path.basename(f)[:-4]}')
        for kid, e in load(f).items():
            tp, _ = pipe_util(e)
            wsched = (e.get('thwarps') or 0) / 4.0
            nm = e['name'].split('<')[0].replace('void ', '')[:33]
            print(f"{nm:<34}{e.get('grid',0):>8}{e.get('dur',0):>9.1f}"
                  f"{e.get('sm',0):>6.1f}{(f'{tp:.1f}' if tp else '-'):>7}"
                  f"{e.get('dram',0):>7.1f}{e.get('l2',0):>6.1f}{e.get('l1',0):>6.1f}"
                  f"{e.get('l2hit',0):>6.1f}"
                  f"{e.get('occ',0):>5.1f}/{e.get('occth',0):<3.0f}{wsched:>8.2f}"
                  f"{e.get('elig',0):>6.2f}{e.get('issue',0):>7.1f}{e.get('waves',0):>7.2f}"
                  f"  {verdict(e)}")
        if a.rules:
            e = list(load(f).values())[0]
            for rn, rd, sp in e['rules']:
                if rn in ('SOLBottleneck', 'HighPipeUtilization', 'CPIStall',
                          'TheoreticalOccupancy', 'Occupancy'):
                    print(f"     [{rn}] {('+'+sp+'% ') if sp else ''}{rd[:190]}")

    print('\nVerdict thresholds: >=80% of peak on a unit = SATURATED, >=60% = leaning,')
    print('otherwise LATENCY-BOUND. <1.0 waves/SM is reported as UNDERFILLED first,')
    print('since a partially-filled grid makes every throughput figure meaningless.')
    print('w/sched = theoretical warps per scheduler; the hardware maximum is 12.')


if __name__ == '__main__':
    sys.exit(main() or 0)
