"""Reference LLM fed llama.cpp's own visual embeddings and positions: does it still write clean text?"""
import glob, json, os, re, sys
import numpy as np, torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
SNAP = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*"))[0]
dump, videos = sys.argv[1], sys.argv[2].split(",")
tok = AutoTokenizer.from_pretrained(SNAP)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(SNAP, torch_dtype=torch.bfloat16, device_map="cuda").eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
out = []
for v in videos:
    j = json.load(open(f"{dump}/{v}.video.json"))
    embd = torch.from_numpy(np.fromfile(f"{dump}/{v}.video.embd.bin", dtype=np.float32).reshape(-1, j["n_embd"])).cuda()
    tokens = np.array(j["tokens"]); vis = torch.from_numpy(tokens < 0).cuda()
    ids = torch.tensor([[int(t) if t >= 0 else 0 for t in tokens]]).cuda()
    pos = torch.tensor(j["pos"]).reshape(-1, 3).T.unsqueeze(1).cuda()       # (3,1,n)
    x = lm.get_input_embeddings()(ids).clone()
    x[0, vis] = embd.to(x.dtype)
    text, past, nxt = [], None, int(j["next_pos"])
    with torch.no_grad():
        o = lm(inputs_embeds=x, position_ids=pos, use_cache=True)
        for _ in range(128):
            logits = model.lm_head(o.last_hidden_state[:, -1])
            t = int(logits.argmax(-1))
            if t in (151645, 151643): break
            text.append(t)
            e = lm.get_input_embeddings()(torch.tensor([[t]]).cuda())
            p = torch.full((3, 1, 1), nxt).cuda(); nxt += 1
            o = lm(inputs_embeds=e, position_ids=p, past_key_values=o.past_key_values, use_cache=True)
    s = tok.decode(text)
    out.append({"video": v, "non_ascii": bool(re.search(r"[^\x00-\x7f]", s)), "n": len(text), "text": s})
    print(json.dumps(out[-1], ensure_ascii=False)[:330])
print("non-ASCII:", sum(o["non_ascii"] for o in out), "of", len(out))
json.dump(out, open(f"{dump}/hf_from_cpp_embd.json", "w"), indent=1, ensure_ascii=False)
