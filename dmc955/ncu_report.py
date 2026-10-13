#!/usr/bin/env python3
"""
Read the grid-aware ncu reports PER (kernel, grid) and state stage coverage.

Two rules from the methodology, enforced here rather than left to the reader:

  * Never average across grid sizes. Each (kernel, grid) is reported on its own
    line with every launch shown, because the same name at a different grid is
    a different kernel with different behaviour.
  * Launch consistency IS the validity check. Launches at one grid agreeing
    within a few percent is evidence; a wide spread means the sample is
    incoherent and the number should not be quoted. Spread is computed and
    anything above --incoherent is flagged, not silently averaged.

A stage-level claim requires stage coverage. The covered share is summed from
the nsys trace and printed per phase, so a claim can be checked against it.
"""
import csv, glob, os, re, sys, argparse, collections
import ncu_targets as T

WANT = [
    ('Duration',                    'dur'),
    ('Compute (SM) Throughput',     'sm%'),
    ('Memory Throughput',           'mem%'),
    ('DRAM Throughput',             'dram%'),
    ('L2 Hit Rate',                 'l2hit%'),
    ('Achieved Occupancy',          'occ%'),
    ('Theoretical Occupancy',       'occ_th%'),
    ('Waves Per SM',                'waves'),
    ('Registers Per Thread',        'regs'),
]

def read_report(path):
    """-> {launch_id: {'grid':int, 'name':str, metrics...}}"""
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith('==')))
    out = collections.OrderedDict()
    for r in rows:
        kid = r.get('ID')
        if kid is None or kid == '':
            continue
        e = out.setdefault(kid, {})
        if 'grid' not in e:
            nums = [int(x) for x in re.findall(r'\d+', r.get('Grid Size', ''))]
            e['grid'] = nums[0] * nums[1] * nums[2] if len(nums) >= 3 else None
            e['name'] = r.get('Kernel Name', '')
        mn, mv, mu = r.get('Metric Name'), r.get('Metric Value'), r.get('Metric Unit')
        for want, key in WANT:
            if mn == want:
                try:
                    v = float(str(mv).replace(',', ''))
                except (TypeError, ValueError):
                    continue
                if key == 'dur':
                    u = (mu or '').lower()
                    v *= 1e3 if u.startswith('m') else (1e-3 if u.startswith('n') else 1)
                e[key] = v
    return out

def spread(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2 or max(vals) == 0:
        return 0.0
    return 100.0 * (max(vals) - min(vals)) / (sum(vals) / len(vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'profiling_results'))
    ap.add_argument('--db', default=T.DB_DEFAULT)
    ap.add_argument('--incoherent', type=float, default=15.0,
                    help='%% spread in Duration above which a sample is flagged')
    a = ap.parse_args()

    K = T.load(a.db); enc, dec = T.phases(K); T.tag(K, enc, dec)

    # Exact stage shares, keyed by the planner's own target identity. Do NOT
    # fuzzy-match on ncu's kernel name: ncu and nsys demangle differently
    # ("(int)1" vs "1"; every cutlass::Kernel2<...> shares a prefix), and a
    # near-match silently attributes one kernel's share to another.
    share = {}
    for ph in ('vit', 'prefill', 'decode'):
        tot, rows = T.histogram(K, ph)
        for name, grid, block, cnt, ns, sh in rows:
            share[(ph, name, grid)] = sh

    targets = {}          # slug -> (phase, nsys name, grids, regex)
    tf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ncu_targets.sh')
    for line in open(tf):
        line = line.strip()
        if not line.startswith('"'):
            continue
        parts = line.strip('"').split('|')
        slug, pat, grids, skip, cnt, pred = parts[:6]
        shapes = []
        if len(parts) > 6 and parts[6]:
            for d in parts[6].split(','):
                g, rg, sm, p50, shr = d.split(':')
                shapes.append(dict(grid=int(g), regs=int(rg), smem=int(sm),
                                   p50=float(p50), share=float(shr)))
        rx = re.compile(pat)
        names = sorted({k['name'] for k in K if rx.search(k['name'])})
        targets[slug] = (slug.split('_')[0],
                         names[0] if len(names) == 1 else None, shapes, pat)

    covered = collections.defaultdict(float)
    print('%-46s %7s %4s %9s %7s %7s %7s %8s %7s %7s  %s'
          % ('kernel (per grid)', 'grid', 'n', 'us', 'SM%', 'DRAM%', 'L2%',
             'occ%/th', 'waves', 'stage%', 'verdict'))
    print('-' * 150)
    cur_phase = None
    for csvf in sorted(glob.glob(os.path.join(a.dir, 'ncu_*.csv'))):
        slug = os.path.basename(csvf)[4:-4]
        ph = slug.split('_')[0]
        if ph not in ('vit', 'prefill', 'decode'):
            continue
        if ph != cur_phase:
            print('\n== %s ==' % ph.upper()); cur_phase = ph
        tgt = targets.get(slug)
        if tgt is None:
            print('  !! %s has no entry in ncu_targets.sh (stale report?)' % slug)
            continue
        _, tname, tshapes, tpat = tgt
        if tname is None:
            print('  !! %s: target regex is no longer unique in the trace' % slug)
        launches = read_report(csvf)
        bygrid = collections.defaultdict(list)
        for e in launches.values():
            bygrid[e['grid']].append(e)

        # A grid can carry two launch shapes (decode grid 3584 is both q_proj at
        # ~29 us and down_proj at ~144 us). Bucketing by grid alone re-mixes them
        # and reports a 134% "spread" that is really two different kernels. Split
        # each grid's measured durations the same way the planner split the trace,
        # then pair measured clusters to trace shapes in order of magnitude --
        # ncu locks clocks, so absolute us differ from the trace but the ORDER and
        # the ratios between shapes are preserved.
        groups = []
        for grid, es in bygrid.items():
            cands = sorted([c for c in tshapes if c['grid'] == grid],
                           key=lambda c: c['p50'])
            if len(cands) < 2:
                groups.append((grid, es, cands[0] if cands else None))
                continue
            cl = T.clusters([e.get('dur', 0.0) for e in es])
            if len(cl) < 2:
                groups.append((grid, es, min(
                    cands, key=lambda c: abs(c['p50'] - (
                        sum(e.get('dur', 0) for e in es) / max(1, len(es)))))))
                continue
            for i, c in enumerate(sorted(cl, key=lambda x: x['p50'])):
                mem = [e for e in es if c['lo'] <= e.get('dur', 0) <= c['hi']]
                groups.append((grid, mem, cands[i] if i < len(cands) else None))

        for grid, es, shape in sorted(groups, key=lambda g: -len(g[1])):
            # attribute to the SHAPE, not just the grid: one grid can carry two
            # problem sizes (decode grid 3584 is both q_proj and down_proj).
            # ncu locks clocks so absolute durations differ from the trace --
            # match on the RATIO between the shapes' medians, not absolute us.
            sh = shape['share'] if shape else None
            sp = spread([e.get('dur') for e in es])
            verdict = 'INCOHERENT (%.0f%% spread)' % sp if sp > a.incoherent else 'consistent'
            if len(es) < 2:
                verdict = 'single launch - no consistency check'
            if sh is not None and sp <= a.incoherent and len(es) >= 2:
                covered[ph] += sh
            def m(k):
                v = [e.get(k) for e in es if e.get(k) is not None]
                return sum(v) / len(v) if v else float('nan')
            nm = es[0]['name'].split('<')[0].replace('void ', '')[:46]
            print('%-46s %7d %4d %9.1f %7.2f %7.2f %7.2f %5.1f/%-4.1f %7.2f %6s  %s'
                  % (nm, grid, len(es), m('dur'), m('sm%'), m('dram%'),
                     m('l2hit%'), m('occ%'), m('occ_th%'), m('waves'),
                     ('%.2f' % sh) if sh is not None else '?', verdict))
            for e in es:
                print('%-46s %7s %4s %9.1f %7.2f %7.2f %7.2f %5.1f'
                      % ('    launch', '', '', e.get('dur', float('nan')),
                         e.get('sm%', float('nan')), e.get('dram%', float('nan')),
                         e.get('l2hit%', float('nan')), e.get('occ%', float('nan'))))

    print('\n' + '=' * 90)
    print('STAGE COVERAGE  (share of stage GPU time backed by a consistent, per-grid sample)')
    for ph in ('vit', 'prefill', 'decode'):
        c = covered[ph]
        note = 'stage-level claims OK' if c >= 70 else 'NOT enough for a stage-level claim'
        print('  %-8s %5.1f%%   %s' % (ph, c, note))

if __name__ == '__main__':
    sys.exit(main() or 0)
