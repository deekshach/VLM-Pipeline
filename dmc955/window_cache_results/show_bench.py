import json, sys
d=json.load(open(sys.argv[1]))
print('native',d.get('native_grid'),[round(x,1) for x in d['native_encode_ms']])
print('grid',d['window_grid'])
for key in ['shapes','shapes_after_alternating']:
    print(key)
    for r in d[key]:
        print(f"{r['w']}x{r['h']} k={r['windows']:2d} prep={r['prep_ms']:.2f} first={r['first_ms']:.1f} steady={r['steady_ms']:.2f}  rest={[round(x,1) for x in r['encode_ms'][1:]]}")
print('alt',[(a['w'],a['h'],round(a['ms'],1)) for a in d['alternating']])
