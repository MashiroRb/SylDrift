import json, sys
import numpy as np, torch, torch.nn.functional as F
npz = np.load(sys.argv[1])
keys = [k for k in npz.files]
cent = F.normalize(torch.tensor(np.stack([npz[k].astype(np.float32).mean(0) for k in keys])), dim=-1)
hit = tot = 0
for i, k in enumerate(keys):
    v = F.normalize(torch.tensor(npz[k].astype(np.float32)), dim=-1)
    pred = (v @ cent.T).argmax(1)
    hit += int((pred == i).sum()); tot += len(v)
rep = {"classes": len(keys), "probes": tot, "top1_acc": round(hit/tot, 4),
       "chance": round(1/len(keys), 5), "ratio": round((hit/tot)/(1/len(keys)), 2)}
print(json.dumps(rep))
json.dump(rep, open(sys.argv[2], "w"), indent=2)
