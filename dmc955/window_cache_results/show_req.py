import json, sys
for fn in sys.argv[1:]:
    d=json.load(open(fn))
    for r in d['requests']:
        print(fn.split('/')[-1], r['video'], 'vit',round(r['vit_ms'],1),'prefill',round(r['prefill_ms'],1))
        for f in r['frames'][1:]:
            print('   ',f['action'],'dirty',f['n_dirty'],'crop',f['crop'],'ms',round(f['encode_ms'],1),
                  'cos mean %.4f min %.3f p05 %.3f reused %.4f fresh %.4f low %.3f'%(f.get('cos_mean',-1),f.get('cos_min',-1),f.get('cos_p05',-1),f.get('cos_reused',-1),f.get('cos_fresh',-1),f.get('frac_below_090',-1)))
