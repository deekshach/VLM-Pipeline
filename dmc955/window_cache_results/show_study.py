import json, sys
import numpy as np
for fn in sys.argv[1:]:
    d=json.load(open(fn))
    st=[s for s in d['studies'] if 'pairs_score_cosmean_cosmin' in s]
    P=np.array([p for s in st for p in s['pairs_score_cosmean_cosmin']])
    print('==',fn.split('/')[-1],'videos',len(st),'grid',st[0]['grid_windows'],'window-pairs',len(P))
    print('  score pctl 10/25/50/75/90:',np.round(np.percentile(P[:,0],[10,25,50,75,90]),1))
    print('  thr  frac_clean  cos_mean(clean) cos_p05(clean) cosmin_p05(clean)')
    for t in [1,2,3,4,6,8,12,16,24]:
        m=P[:,0]<=t
        if m.sum()==0: print(f'  {t:3d}  0'); continue
        print(f'  {t:3d}  {m.mean():.3f}       {P[m,1].mean():.4f}        {np.percentile(P[m,1],5):.4f}       {np.percentile(P[m,2],5):.4f}')
    print('  all windows: cos mean %.4f'%P[:,1].mean())
    for key in ['crops','mosaics']:
        agg={}
        for s in st:
            for c in s[key]:
                k=(str(c['windows']), c.get('on_grid','-'))
                agg.setdefault(k,[]).append((c['mean'],c['p05'],c['min']))
        print(' ',key,'(windows, on_grid): mean / p05 / min averaged over videos')
        for k,v in agg.items():
            v=np.array(v); print('   ',k, np.round(v.mean(0),4))
