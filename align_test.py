import sys, json
from funasr import AutoModel
m = AutoModel(model="fa-zh", disable_update=True, device="cpu")
p, txt = sys.argv[1], sys.argv[2]
try:
    res = m.generate(input=[(txt, p)], data_type="sound")
    print("OK", json.dumps(res, ensure_ascii=False)[:700])
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:200])
