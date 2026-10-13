#!/usr/bin/env python3
"""
Grid-aware ncu target planner.

A kernel name is not a kernel: the same name is launched at several grid sizes
that behave completely differently, and averaging across them describes no
actual launch. This derives ncu --launch-skip/--launch-count values that land
on a CHOSEN (kernel, grid) from an existing nsys trace, and predicts the grid
of every launch it will profile so the resulting report can be verified.

  plan     emit the target table (shell-sourceable with --sh)
  window   emit a whole-repeating-unit window per phase (all kernels, no -k)
  verify   check an ncu report's per-launch grids against the prediction

Launch numbering follows CPU launch order (ncu's order), reconstructed by
joining CUPTI_ACTIVITY_KIND_KERNEL to _RUNTIME on correlationId; kernels
launched from CUDA graph nodes have no runtime record and fall back to GPU
start time.
"""
import sqlite3, sys, os, re, csv, collections, argparse

DB_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'profiling_results', 'handshake_7B.sqlite')
FRAME_KERNELS = 1270      # ViT kernels per frame, stream 17
N_FRAMES      = 13

def load(db):
    c = sqlite3.connect(db)
    sid = {i: s for i, s in c.execute('select id,value from StringIds')}
    launch = {}
    for corr, st in c.execute(
            'select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME'):
        launch[corr] = st
    K = []
    for (s, e, stream, nid, gx, gy, gz, bx, by, bz, regs, ssm, dsm,
         corr, gnode) in c.execute('''
            select start,end,streamId,demangledName,gridX,gridY,gridZ,
                   blockX,blockY,blockZ,registersPerThread,
                   staticSharedMemory,dynamicSharedMemory,correlationId,graphNodeId
            from CUPTI_ACTIVITY_KIND_KERNEL'''):
        K.append(dict(gpu_start=s, gpu_end=e, dur=e - s, stream=stream,
                      name=sid[nid], grid=gx * gy * gz, block=bx * by * bz,
                      regs=regs, smem=ssm + dsm,
                      order_key=launch.get(corr, s), graph=gnode is not None))
    # CPU launch order == ncu's launch numbering
    K.sort(key=lambda k: (k['order_key'], k['gpu_start']))
    for i, k in enumerate(K):
        k['idx'] = i
    return K

def phases(K):
    """encode/decode windows, from the ViT stream's per-frame structure."""
    s17 = [k for k in K if k['stream'] == 17]
    s17.sort(key=lambda k: k['gpu_start'])
    nreq = len(s17) // (FRAME_KERNELS * N_FRAMES)
    enc, dec = [], []
    for r in range(nreq):
        enc.append((s17[r * FRAME_KERNELS * N_FRAMES]['gpu_start'],
                    s17[(r + 1) * FRAME_KERNELS * N_FRAMES - 1]['gpu_end']))
    gemv = sorted([k for k in K if k['stream'] == 15
                   and 'mul_mat_vec_f' in k['name']],
                  key=lambda k: k['gpu_start'])
    end = max(k['gpu_end'] for k in K)
    for i, (a, b) in enumerate(enc):
        ds = next((k['gpu_start'] for k in gemv if k['gpu_start'] >= b), b)
        de = enc[i + 1][0] if i + 1 < len(enc) else end
        dec.append((ds, de))
    return enc, dec

def tag(K, enc, dec):
    """
    vit     = stream 17 inside an encode window
    prefill = stream 15 from the encode start until decode begins. NOT until
              the last ViT kernel: the final frame's prefill runs after the
              vision tower has finished, and cutting there drops it.
    decode  = stream 15 from the first decode GEMV to the next encode.
    """
    for k in K:
        k['phase'] = None
    for k in K:
        k['pinst'] = None
    for i, (a, b) in enumerate(enc):
        dstart = dec[i][0]
        for k in K:
            if k['stream'] == 17 and a <= k['gpu_start'] <= b:
                k['phase'], k['pinst'] = 'vit', i
            elif k['stream'] == 15 and a <= k['gpu_start'] < dstart:
                k['phase'], k['pinst'] = 'prefill', i
            elif k['stream'] == 15 and dec[i][0] <= k['gpu_start'] <= dec[i][1]:
                k['phase'], k['pinst'] = 'decode', i
    return K


def histogram(K, phase, top=99):
    sel = [k for k in K if k['phase'] == phase]
    tot = sum(k['dur'] for k in sel)
    agg = collections.defaultdict(lambda: [0, 0])
    for k in sel:
        a = agg[(k['name'], k['grid'], k['block'])]
        a[0] += 1; a[1] += k['dur']
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    return tot, [(n, g, b, cnt, ns, 100.0 * ns / tot)
                 for (n, g, b), (cnt, ns) in rows[:top]]

def plan_target(K, name, grid, phase, count=3, prefer='mid'):
    """
    Choose -s/-c for (name, grid). ncu -k regex matches by name only, so the
    window is a run of consecutive same-NAME launches; we pick the run whose
    members all have the wanted grid where possible, and always report the
    predicted grid of each launch so the report can be verified.
    """
    matches = [k for k in K if k['name'] == name]           # ncu -k order
    for j, k in enumerate(matches):
        k['mi'] = j
    cand = [k for k in matches if k['grid'] == grid and k['phase'] == phase]
    if not cand:
        return None
    # longest run of consecutive match-indices that all have the target grid
    runs, cur = [], [cand[0]]
    for k in cand[1:]:
        if k['mi'] == cur[-1]['mi'] + 1:
            cur.append(k)
        else:
            runs.append(cur); cur = [k]
    runs.append(cur)
    usable = [r for r in runs if len(r) >= count] or runs
    # mid-phase: prefill attention grows with context, early launches are small
    pick = usable[len(usable) // 2] if prefer == 'mid' else usable[0]
    if len(pick) >= count:
        off = (len(pick) - count) // 2
        chosen = pick[off:off + count]
    else:
        chosen = pick
    skip = chosen[0]['mi']
    n = len(chosen)
    pred = [matches[skip + i]['grid'] for i in range(n)]
    pure = all(p == grid for p in pred)
    return dict(name=name, grid=grid, phase=phase, skip=skip, count=n,
                predicted=pred, pure=pure,
                runlen=len(pick), n_at_grid=len(cand),
                mean_us=sum(k['dur'] for k in chosen) / n / 1e3)

def clusters(durs, ratio=1.6, min_frac=0.04):
    """
    Split a set of launch durations into distinct clusters.

    (kernel, grid) is NOT always enough to identify a kernel: llama.cpp launches
    q_proj (K=3584) and down_proj (K=18944) at the SAME grid, same registers and
    same shared memory, differing only in the weight matrix they stream -- 25.7 MB
    vs 136 MB, which at ~1008 GB/s is ~25 us vs ~135 us. nsys records no problem
    size, so the only trace-visible discriminator is the duration itself.

    Split on a large relative gap in the sorted durations, requiring both sides
    to be a meaningful share of the launches so that a single slow outlier does
    not manufacture a cluster.
    """
    d = sorted(durs)
    if len(d) < 4:
        return ([dict(lo=d[0], hi=d[-1], n=len(d), p10=d[0], p25=d[0],
                      p50=d[len(d) // 2], p75=d[-1], p90=d[-1])] if d else [])
    cuts = []
    for i in range(1, len(d)):
        if d[i - 1] > 0 and d[i] / d[i - 1] >= ratio:
            if min(i, len(d) - i) >= max(2, int(min_frac * len(d))):
                cuts.append(i)
    def pct(seg, q):
        return seg[min(len(seg) - 1, max(0, int(q * (len(seg) - 1))))]
    out, lo = [], 0
    for c in cuts + [len(d)]:
        seg = d[lo:c]
        if seg:
            out.append(dict(lo=seg[0], hi=seg[-1], n=len(seg),
                            p10=pct(seg, 0.10), p25=pct(seg, 0.25),
                            p50=pct(seg, 0.50), p75=pct(seg, 0.75),
                            p90=pct(seg, 0.90)))
        lo = c
    return out


def trend(pairs):
    """Spearman rho between launch order and duration -- separates 'two distinct
    kernels sharing a grid' (no trend, interleaved) from 'one kernel whose cost
    grows with context' (strong trend), which need different sampling."""
    if len(pairs) < 8:
        return 0.0
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def launch_shapes(K, phase, top=14):
    """Distinct launch shapes in a phase: (name, grid, regs, smem, duration cluster)."""
    sel = [k for k in K if k['phase'] == phase]
    tot = sum(k['dur'] for k in sel)
    groups = collections.defaultdict(list)
    for k in sel:
        groups[(k['name'], k['grid'], k['regs'], k['smem'])].append(k)
    out = []
    for (name, grid, regs, smem), ks in groups.items():
        ks.sort(key=lambda k: k['idx'])
        cl = clusters([k['dur'] / 1e3 for k in ks])
        for c in cl:
            lo, hi, n = c['lo'], c['hi'], c['n']
            mem = [k for k in ks if lo <= k['dur'] / 1e3 <= hi]
            # per REQUEST: context grows within a request and resets between
            # them, so a rho taken across both is a sawtooth and reads ~0
            byreq = collections.defaultdict(list)
            for k in mem:
                byreq[k['pinst']].append((k['idx'], k['dur']))
            rhos = [trend(v) for v in byreq.values() if len(v) >= 8]
            rho = max(rhos, key=abs) if rhos else 0.0
            ns = sum(k['dur'] for k in mem)
            out.append(dict(name=name, grid=grid, regs=regs, smem=smem,
                            lo=lo, hi=hi, n=n, ms=ns / 1e6,
                            p10=c['p10'], p25=c['p25'], p50=c['p50'],
                            p75=c['p75'], p90=c['p90'],
                            share=100.0 * ns / tot, rho=rho,
                            multimodal=len(cl) > 1,
                            grows=abs(rho) > 0.7 and hi / max(lo, 1e-9) > 1.3))
    out.sort(key=lambda d: -d['share'])
    return tot, out[:top]


def cmd_shapes(a):
    K = load(a.db); enc, dec = phases(K); tag(K, enc, dec)
    for phase in a.phases:
        tot, rows = launch_shapes(K, phase, a.top)
        print('=' * 118)
        print('PHASE %-8s  %.1f ms over the trace -- distinct LAUNCH SHAPES, '
              'not kernel names' % (phase, tot / 1e6))
        print('%6s %7s %5s %7s %9s %9s %9s %9s %6s  %s'
              % ('share%', 'grid', 'regs', 'smem', 'n', 'us_p10', 'us_p50',
                 'us_p90', 'rho', 'kernel / note'))
        for r in rows:
            note = ''
            if r['multimodal']:
                note = '  <- shares its grid with another problem size'
            if r['grows']:
                note = '  <- grows with context (rho=%.2f): sample at a STATED position' % r['rho']
            if not note and r['p90'] / max(r['p10'], 1e-9) > 1.5:
                note = ('  <- %.0fx spread at one grid: interleaved problem '
                        'sizes, target the median' % (r['p90'] / max(r['p10'], 1e-9)))
            print('%6.2f %7d %5d %7d %9d %9.1f %9.1f %9.1f %6.2f  %s%s'
                  % (r['share'], r['grid'], r['regs'], r['smem'], r['n'],
                     r['p10'], r['p50'], r['p90'], r['rho'],
                     label(r['name'])[:40], note))
        print()


def plan_grouped(K, name, shapes, per_grid=3, phase=None, warn_window=96):
    """
    One ncu window per (kernel name, phase) that captures >= per_grid IN-PHASE
    launches of EACH wanted grid, minimising the window length in ncu's own
    launch numbering -- because that length, not the number of useful launches,
    is what ncu actually replays and pays for.

    ncu -k matches by name only, so a window that mixes grids is free: it costs
    one run instead of several, and it supplies the per-grid launch-consistency
    check that a single profiled launch cannot give. Every launch's grid is
    predicted so the report is read per (kernel, grid), never as a mean across
    them -- averaging across grids describes no actual launch.

    Biased toward mid-phase: -c N alone takes the FIRST N matches, which are
    unrepresentative wherever work grows with context (prefill attention).
    """
    allm = [k for k in K if k['name'] == name]
    for j, k in enumerate(allm):
        k['mi'] = j                      # index among same-name launches == ncu's
    if not allm:
        return None
    want = {i: per_grid for i in range(len(shapes))}

    def good(k, i):
        sh = shapes[i]
        if k['grid'] != sh['grid'] or k['regs'] != sh['regs'] or k['smem'] != sh['smem']:
            return False
        if phase is not None and k['phase'] != phase:
            return False
        # stay inside the duration cluster: where several problem sizes share a
        # grid they are only separable by duration, and a window that mixes them
        # yields an incoherent sample that cannot be quoted
        us = k['dur'] / 1e3
        # where one grid carries a wide spread of problem sizes, a p10..p90 band
        # is still wide enough to mix them; sample the interquartile core so the
        # profiled launches are representative of the shape's median cost
        wide = sh['p90'] > 1.5 * max(sh['p10'], 1e-9)
        lo_b, hi_b = ((sh['p25'], sh['p75']) if wide else (sh['p10'], sh['p90']))
        return lo_b * 0.92 <= us <= hi_b * 1.08

    n, mid = len(allm), None
    inph = [k['mi'] for k in allm
            if any(good(k, i) for i in range(len(shapes)))]
    if not inph:
        return None
    mid = inph[len(inph) // 2]

    have = collections.Counter()
    best, lo = None, 0
    for hi in range(n):
        for i in range(len(shapes)):
            if good(allm[hi], i):
                have[i] += 1
        while lo <= hi and all(have[i] >= want[i] for i in range(len(shapes))):
            w = hi - lo + 1
            key = (w, abs((lo + hi) // 2 - mid))
            if best is None or key < best[0]:
                best = (key, lo, w)
            for i in range(len(shapes)):
                if good(allm[lo], i):
                    have[i] -= 1
            lo += 1
    if not best:
        return None
    _, s0, w = best
    win = allm[s0:s0 + w]
    useful = {shapes[i]['grid']: [k for k in win if good(k, i)]
              for i in range(len(shapes))}
    return dict(name=name, grids=[sh['grid'] for sh in shapes], skip=s0,
                count=w, phase=phase,
                predicted=[k['grid'] for k in win],
                per_grid={g: len(v) for g, v in useful.items()},
                us={g: (sum(k['dur'] for k in v) / max(1, len(v)) / 1e3)
                    for g, v in useful.items()},
                graph=any(k['graph'] for k in win),
                costly=w > warn_window)


def minimal_regex(name, all_names):
    r"""
    Shortest ANCHORED prefix of the demangled name that is unique among every
    kernel name in the trace. Full mangled signatures are unwieldy, and
    re.escape() escapes spaces as '\ ', which ncu's regex flavour rejects --
    escape only genuine metacharacters.
    """
    META = set(r'\^$.|?*+()[]{}')
    def esc(t):
        return ''.join('\\' + ch if ch in META else ch for ch in t)
    others = [n for n in all_names if n != name]
    # prefer cutting at a structural boundary, then fall back to any length
    cuts = [i for i, ch in enumerate(name) if ch in '<,('] + [len(name)]
    for c in sorted(set(cuts)):
        pre = name[:c + 1] if c < len(name) else name
        if not any(o.startswith(pre) for o in others):
            return '^' + esc(pre)
    return '^' + esc(name)



def label(name):
    """Short but DISCRIMINATING: several cutlass/mul_mat_f instantiations share
    everything up to the first '<', so keep a distinguishing tail."""
    m = re.search(r'<(.+)>', name)
    head = name.split('<')[0].replace('void ', '')
    if 'cutlass' in name:
        inner = re.search(r'cutlass_\w+', name)
        return inner.group(0) if inner else head
    if m:
        return '%s<%s>' % (head, m.group(1)[:44])
    return head


def plan_window(K, phase, unit_layers=1):
    """
    A contiguous ALL-KERNEL window covering one repeating unit of a phase
    (one ViT layer / one prefill layer / one decode token), taken mid-phase.
    Profiled with no -k, this covers ~100% of the phase's kernel mix instead
    of a hand-picked name list. Requires the window to be contiguous in ncu
    launch order, which it only is if no other stream interleaves -- reported.
    """
    sel = [k for k in K if k['phase'] == phase]
    if not sel:
        return None
    sel.sort(key=lambda k: k['idx'])
    # repeating unit: ViT 32 layers/frame, prefill 28 layers/frame, decode 28/token
    if phase == 'vit':
        per_unit, n_units = FRAME_KERNELS // 32, 32 * N_FRAMES
    elif phase == 'prefill':
        per_unit = max(1, len(sel) // (28 * N_FRAMES * 2)); n_units = 28 * N_FRAMES
    else:
        per_unit = max(1, len(sel) // (128 * 2 * 28)); n_units = 128 * 28
    per_unit *= unit_layers
    mid = len(sel) // 2
    win = sel[mid:mid + per_unit]
    lo, hi = win[0]['idx'], win[-1]['idx']
    contiguous = (hi - lo + 1) == len(win)
    foreign = [k for k in K if lo <= k['idx'] <= hi and k['phase'] != phase]
    tot = sum(k['dur'] for k in sel)
    return dict(phase=phase, skip=lo, count=hi - lo + 1, n_wanted=len(win),
                contiguous=contiguous, n_foreign=len(foreign),
                window_ms=sum(k['dur'] for k in win) / 1e6,
                phase_ms=tot / 1e6,
                distinct=len({(k['name'], k['grid']) for k in win}))

# ---- top targets, chosen by (kernel, grid) time-dominance ------------------
def top_targets(K, phase, want_cov=70.0, max_n=8):
    """Target launch SHAPES by time-dominance: (name, grid, regs, smem, duration
    cluster). Grid alone is not enough -- q_proj and down_proj share grid 3584."""
    tot, rows = launch_shapes(K, phase, top=64)
    out, cum = [], 0.0
    for r in rows:
        if len(out) >= max_n or cum >= want_cov:
            break
        out.append(r); cum += r['share']
    return out, cum


def cmd_plan(a):
    K = load(a.db); enc, dec = phases(K); tag(K, enc, dec)
    # collect (name, grid) targets per phase, then merge by name: one ncu run
    # can serve several grids AND several phases of the same kernel name.
    per_phase, byname = {}, collections.OrderedDict()
    for phase in a.phases:
        tgts, cov = top_targets(K, phase, a.coverage, a.max_targets)
        per_phase[phase] = (tgts, cov)
        for r in tgts:
            byname.setdefault(r['name'], [])
            byname[r['name']].append((r, phase))

    plans = {}
    for name, gs in byname.items():
        for ph in {p for _, p in gs}:
            shapes = [r for r, p in gs if p == ph]
            plans[(name, ph)] = plan_grouped(K, name, shapes, a.count, phase=ph)

    for phase in a.phases:
        tgts, cov = per_phase[phase]
        runs = {n for n, gs in byname.items() if any(p == phase for _, p in gs)}
        print('=' * 108)
        print('PHASE %-8s  target coverage %.1f%% of stage GPU time  '
              '(%d shapes, %d ncu runs)' % (phase, cov, len(tgts), len(runs)))
        got = 0.0
        for name, gs in byname.items():
            mine = [(r['grid'], r['share'], r) for r, p in gs if p == phase]
            if not mine:
                continue
            p = plans[(name, phase)]
            if not p:
                print('  !! no window found for', label(name)[:80]); continue
            note = ''
            if p['costly']:
                note += '  [WIDE: ncu replays %d launches]' % p['count']
            if p['graph']:
                note += '  [graph nodes: needs --graph-profiling node]'
            print('  %-58s  -s %-6d -c %-4d%s'
                  % (label(name)[:58], p['skip'], p['count'], note))
            for g, share, r in mine:
                got += share
                print('      grid %-7d regs %-4d smem %-6d %5.2f%% of stage   '
                      'n=%d in window   %7.1f us (trace p50 %.1f)'
                      % (g, r['regs'], r['smem'], share,
                         p['per_grid'].get(g, 0), p['us'].get(g, float('nan')),
                         r['p50']))

        print('  --> achievable coverage this phase: %.1f%%' % got)
        print()

    if a.sh:
        with open(a.sh, 'w') as f:
            f.write('# generated by ncu_targets.py plan -- do not hand-edit\n')
            f.write('# slug|name-regex|grids|-s|-c|predicted-grids|'
                    'shapes(grid:regs:smem:p50us:share%)\n')
            f.write('NCU_TARGETS=(\n')
            used = set()
            all_names = {k['name'] for k in K}
            for (name, ph), p in plans.items():
                if not p:
                    continue
                slug = '%s_%s' % (ph, re.sub(r'[^a-z0-9]+', '_',
                                  label(name).lower()).strip('_')[:36])
                base, i = slug, 2
                while slug in used:
                    slug = '%s%d' % (base, i); i += 1
                used.add(slug)
                shapes = [r for r, ph2 in byname[name] if ph2 == ph]
                desc = ','.join('%d:%d:%d:%.3f:%.4f'
                                % (r['grid'], r['regs'], r['smem'],
                                   r['p50'], r['share']) for r in shapes)
                f.write('  "%s|%s|%s|%d|%d|%s|%s"\n'
                        % (slug, minimal_regex(name, all_names),
                           ','.join(map(str, p['grids'])),
                           p['skip'], p['count'],
                           ','.join(map(str, p['predicted'])), desc))
            f.write(')\n')
        print('wrote %s (%d ncu runs)' % (a.sh, len(used)))

def cmd_window(a):
    K = load(a.db); enc, dec = phases(K); tag(K, enc, dec)
    print('%-8s %8s %8s %6s %10s %9s %s'
          % ('phase', '-s', '-c', 'shapes', 'window_ms', 'phase_ms', 'contiguous'))
    for phase in a.phases:
        w = plan_window(K, phase, a.layers)
        if not w:
            continue
        print('%-8s %8d %8d %6d %10.2f %9.1f  %s%s'
              % (phase, w['skip'], w['count'], w['distinct'], w['window_ms'],
                 w['phase_ms'], w['contiguous'],
                 '' if w['contiguous'] else '  (%d foreign launches interleave)' % w['n_foreign']))

def cmd_verify(a):
    """Compare an ncu --csv details export against the predicted grids."""
    got = []
    with open(a.csv) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith('=')))
    seen = {}
    for r in rows:
        kid = r.get('ID')
        nm = r.get('Kernel Name', '')
        if kid is None or kid in seen:
            continue
        seen[kid] = True
        gs = r.get('Grid Size') or ''
        got.append((kid, nm[:50], gs))
    # tolerate a caller that passed more pipe-separated fields than it meant to:
    # the predicted-grid list is the leading field
    raw = (a.predicted or '').split('|')[0]
    pred = [int(x) for x in raw.split(',') if x.strip().isdigit()]
    print('%4s %12s %12s  %s' % ('ID', 'grid_actual', 'grid_pred', 'kernel'))
    ok = True
    for i, (kid, nm, gs) in enumerate(got):
        nums = [int(x) for x in re.findall(r'\d+', gs)]
        g = str(nums[0] * nums[1] * nums[2]) if len(nums) >= 3 else ''
        p = pred[i] if i < len(pred) else ''
        match = (str(p) == g) if p != '' else None
        if match is False:
            ok = False
        print('%4s %12s %12s  %s %s' % (kid, g, p, nm,
              '' if match is not False else '  <-- MISMATCH'))
    print('\nVERDICT:', 'grids match prediction' if ok else
          'PREDICTION DRIFTED -- launch numbering differs from the nsys trace; '
          're-plan against a trace of this exact build/config')
    return 0 if ok else 1

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', default=DB_DEFAULT)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan');  p.set_defaults(fn=cmd_plan)
    p.add_argument('--phases', nargs='+', default=['vit', 'prefill', 'decode'])
    p.add_argument('--coverage', type=float, default=70.0)
    p.add_argument('--max-targets', type=int, default=8)
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--sh', help='write a shell-sourceable target table here')
    p = sub.add_parser('shapes'); p.set_defaults(fn=cmd_shapes)
    p.add_argument('--phases', nargs='+', default=['vit', 'prefill', 'decode'])
    p.add_argument('--top', type=int, default=14)
    p = sub.add_parser('window'); p.set_defaults(fn=cmd_window)
    p.add_argument('--phases', nargs='+', default=['vit', 'prefill', 'decode'])
    p.add_argument('--layers', type=int, default=1)
    p = sub.add_parser('verify'); p.set_defaults(fn=cmd_verify)
    p.add_argument('csv'); p.add_argument('--predicted', default='')
    a = ap.parse_args()
    sys.exit(a.fn(a) or 0)
