"""简芯 v2 — 融合版
融合：蒸馏(过滤层) + 海马体(时间衰减) + 锚点(重要性加权) + 长上下文(搜索增强)
外部U盘：知识图谱(独立搜索，结果合并)
"""
import sys, os, glob, json, time, threading, math, random, re, struct
from collections import Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

_当前目录 = os.path.dirname(os.path.abspath(__file__))
_d = os.path.join(_当前目录, '中枢数据')
os.makedirs(_d, exist_ok=True)

# U盘组件（独立可拔出）
try: from slot_海马体 import 海马体
except: 海马体 = None
try: from slot_锚点 import 锚点记忆
except: 锚点记忆 = None
try: from slot_蒸馏 import 知识蒸馏器
except: 知识蒸馏器 = None
try: from slot_上下文 import 长上下文
except: 长上下文 = None
try: from slot_图谱 import 知识图谱
except: 知识图谱 = None

# ====== 神经网络核心 ======
def _sf(v):
    mx = max(v); ex = [math.exp(x - mx) for x in v]; s = sum(ex)
    return [(x) / (s + 1e-10) for x in ex]

class TextVectorizer:
    def __init__(self, mf=256):
        self.vocab = {}; self.idf = {}; self.max_features = mf
    def tokenize(self, text):
        t = []; t.extend(re.findall(r'[\u4e00-\u9fff]', text))
        t.extend(re.findall(r'[a-zA-Z0-9_]+', text.lower())); return t
    def fit(self, texts):
        at = Counter(); df = Counter()
        for txt in texts:
            for t in set(self.tokenize(txt)): df[t] += 1
            at.update(self.tokenize(txt))
        top = [t for t, _ in at.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(top)}
        nd = len(texts)
        for t in self.vocab: self.idf[t] = math.log((nd + 1) / (df.get(t, 0) + 1)) + 1
    def transform(self, text):
        v = [0.0] * self.max_features
        tokens = self.tokenize(text); tf = Counter(tokens)
        mtf = max(tf.values()) if tf else 1
        for t, c in tf.items():
            if t in self.vocab:
                i = self.vocab[t]
                if i < self.max_features: v[i] = (c / mtf) * self.idf.get(t, 1.0)
        return v
    def search(self, query, texts, top_k=10):
        vq = self.transform(query)
        scored = []
        for item in texts:
            text = item if isinstance(item, str) else item.get('text', str(item))
            vd = self.transform(text)
            score = sum(a * b for a, b in zip(vq, vd))
            scored.append((text, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

class 简芯:
    def __init__(self, ins=256, hid=None, outs=16, lr=0.05):
        if hid is None: hid = [64, 32]
        self.layers = []; sizes = [ins] + hid + [outs]
        for i in range(len(sizes) - 1):
            sc = math.sqrt(2.0 / max(1, sizes[i] + sizes[i + 1]))
            w = [[random.gauss(0, sc) for _ in range(sizes[i + 1])] for _ in range(sizes[i])]
            self.layers.append({'w': w, 'b': [0.0] * sizes[i + 1], 'ni': sizes[i], 'no': sizes[i + 1]})
        self.lr = lr; self.vec = TextVectorizer(mf=ins)
        self.标签映射 = {}; self.反向标签 = {}
    def forward(self, x):
        for l in self.layers:
            z = [sum(x[i] * l['w'][i][j] for i in range(l['ni'])) + l['b'][j] for j in range(l['no'])]
            x = [max(0, v) for v in z]
        return x
    def 投喂(self, 文本, 来源):
        tid = self.标签映射.get(来源)
        if tid is None:
            tid = len(self.标签映射)
            self.标签映射[来源] = tid
            self.反向标签[tid] = 来源
        x = self.vec.transform(文本)
        y = [1.0 if i == tid else 0.0 for i in range(self.layers[-1]['no'])]
        sm = _sf(self.forward(x))
        delta = [o - t for o, t in zip(sm, y)]
        for idx in range(len(self.layers) - 1, -1, -1):
            l = self.layers[idx]
            for j in range(l['no']):
                dj = delta[j]
                for i in range(l['ni']): l['w'][i][j] -= self.lr * dj * x[i]
                l['b'][j] -= self.lr * dj
            if idx > 0:
                pd = [sum(delta[j] * l['w'][i][j] for j in range(l['no'])) for i in range(l['ni'])]
                delta = pd
        return {'已学习': True}
    def 批量训练(self, 文本列表, 来源列表, 轮次=10):
        if not 文本列表: return
        self.vec.fit(文本列表)
        for 文本 in set(文本列表): self.投喂(文本, 来源列表[文本列表.index(文本)])
    def 预测(self, 文本):
        x = self.vec.transform(文本); sm = _sf(self.forward(x)); mx = max(sm)
        cats = [self.反向标签.get(i, f'类{i}') for i in range(len(sm))]
        return {'来源': cats[sm.index(mx)], '置信度': round(mx, 4)}
    def 保存(self, 路径):
        import pickle
        with open(路径, 'wb') as f: pickle.dump({
            'vec': self.vec, '标签映射': self.标签映射,
            '反向标签': {str(k): v for k, v in self.反向标签.items()},
            'lr': self.lr, 'layers': [{'w': l['w'], 'b': l['b']} for l in self.layers],
        }, f)
    def 加载(self, 路径):
        import pickle
        with open(路径, 'rb') as f:
            d = pickle.load(f)
        self.vec = d.get('vec', self.vec)
        self.标签映射 = d.get('标签映射', {})
        self.反向标签 = {int(k): v for k, v in d.get('反向标签', {}).items()}
        self.lr = d.get('lr', self.lr)
        for i, ld in enumerate(d.get('layers', [])):
            if i < len(self.layers): self.layers[i]['w'] = ld['w']; self.layers[i]['b'] = ld['b']

# ====== 融合守护 ======
class 简芯守护:
    def __init__(self):
        模型路径 = os.path.join(_d, '简芯权重.safetensors')
        self.网络 = 简芯(ins=256, hid=[64, 32], outs=16, lr=0.01)
        if os.path.exists(模型路径):
            try: self.网络.加载(模型路径); print('[简芯] 加载已有权重')
            except: pass
        self.条目 = []  # 记忆条目
        self.蒸馏器 = None
        if 知识蒸馏器:
            try: self.蒸馏器 = 知识蒸馏器(None)
            except: pass
        self.海马体 = 海马体() if 海马体 else None
        self.锚点 = 锚点记忆() if 锚点记忆 else None
        self.上下文 = 长上下文() if 长上下文 else None
        self.图谱 = None
        if 知识图谱:
            try: self.图谱 = 知识图谱(None)
            except: pass
        self.已学文件 = {}; self.总计数 = 0; self.运行中 = True

    def 投喂(self, 文本, 来源='对话'):
        if not isinstance(文本, str) or len(文本.strip()) <= 2: return {'已过滤': True}
        # 蒸馏过滤
        for e in self.条目:
            sa, sb = set(文本), set(e['text'])
            if sa and sb and len(sa & sb) / len(sa | sb) > 0.85:
                return {'已过滤': True}
        self.网络.投喂(文本, 来源)
        self.条目.append({'text': 文本, 'source': 来源, 'time': time.time()})
        self.总计数 += 1
        # 海马体
        if self.海马体:
            try: self.海马体.投喂(文本)
            except: pass
        # 上下文
        if self.上下文:
            try: self.上下文.喂(文本)
            except: pass
        # 锚点自动标记
        if self.锚点 and any(kw in 文本 for kw in ['关键', '重要', '记住', '方案', '核心', '部署', '策略', '架构', '规则']):
            try: self.锚点.钉(文本)
            except: pass
        if self.总计数 % 5 == 0: self.持久化()
        return {'已学习': True, '总计数': self.总计数}

    def 搜索(self, 关键词, top_k=10):
        if not 关键词 or not 关键词.strip(): return []
        # 上下文增强
        q = 关键词
        if self.上下文:
            try:
                ctx = self.上下文.获取(3)
                if ctx: q = 关键词 + ' ' + ''.join(ctx)[:100]
            except: pass
        # TF-IDF 搜索
        结果 = self.网络.vec.search(q, [e['text'] for e in self.条目], top_k * 2)
        # 时间衰减
        now = time.time()
        衰减后 = []
        for t, s in 结果:
            ts = next((e['time'] for e in self.条目 if e['text'] == t), now)
            衰减 = math.exp(-(now - ts) / 86400)
            衰减后.append((t, s * (0.5 + 0.5 * 衰减)))
        结果 = 衰减后
        # 锚点加权
        if self.锚点:
            try:
                al = [a.get('text', '') for a in (self.锚点.搜('') or [])]
                if al:
                    for i, (t, s) in enumerate(结果):
                        for a in al:
                            sa, sb = set(t), set(a)
                            if sa and sb and len(sa & sb) / len(sa | sb) > 0.7:
                                结果[i] = (t, s * 1.2); break
            except: pass
        # 图谱合并
        if self.图谱:
            try:
                for g in (self.图谱.搜(关键词) or []):
                    gt = g.get('text', g.get('name', str(g)))
                    gs = g.get('score', g.get('weight', 0.5))
                    结果.append((gt, gs))
            except: pass
        # 去重排序
        seen = {}
        for t, s in 结果:
            if t in seen: seen[t] = max(seen[t], s)
            else: seen[t] = s
        排序 = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{'text': t, 'score': round(s, 4)} for t, s in 排序]

    def 预测(self, 文本):
        return self.网络.预测(文本)

    def 扫描记忆(self):
        根 = os.path.abspath(os.path.join(_当前目录, '..', '..'))
        文件 = []
        for n in ['MEMORY.md', 'SOUL.md', 'USER.md', 'AGENTS.md', 'IDENTITY.md', 'TOOLS.md', 'HEARTBEAT.md']:
            p = os.path.join(根, n)
            if os.path.exists(p): 文件.append((n, p))
        for p in glob.glob(os.path.join(根, 'memory', '*.md')):
            文件.append((os.path.basename(p), p))
        texts, sources = [], []
        for 名, 路径 in 文件:
            try:
                mt = os.path.getmtime(路径)
                if 路径 not in self.已学文件 or mt > self.已学文件[路径]:
                    with open(路径, encoding='utf-8') as f:
                        for seg in [s.strip() for s in f.read().split('\n\n') if len(s.strip()) > 20]:
                            texts.append(seg)
                            if 名 in ('MEMORY.md', 'SOUL.md', 'USER.md', 'AGENTS.md', 'IDENTITY.md', 'TOOLS.md', 'HEARTBEAT.md'):
                                sources.append('身份记忆')
                            elif 名.startswith('task-summary_'): sources.append('任务记录')
                            else: sources.append('日常记忆')
                    self.已学文件[路径] = mt
            except: pass
        if texts:
            self.网络.批量训练(texts, sources, 轮次=5)
            self.总计数 += len(texts)
            for t, s in zip(texts, sources): self.条目.append({'text': t, 'source': s, 'time': time.time()})
            self.持久化()
        return len(texts)

    def 持久化(self):
        try: self.网络.保存(os.path.join(_d, '简芯权重.safetensors'))
        except: pass

    def 启动(self):
        新 = self.扫描记忆(); self.持久化()
        def _循环():
            while self.运行中: time.sleep(300); self.扫描记忆()
        threading.Thread(target=_循环, daemon=True).start()
        try:
            while self.运行中: time.sleep(10)
        except: self.运行中 = False; self.持久化()

# ====== HTTP 守护 ======
class Handler(BaseHTTPRequestHandler):
    def _j(self, data, s=200):
        self.send_response(s)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    def do_GET(self):
        p = urlparse(self.path).path.rstrip('/')
        q = {k: v for k, v in parse_qs(urlparse(self.path).query).items()}
        if not hasattr(self.server, '守护'): self.server.守护 = 简芯守护(); self.server.守护.扫描记忆()
        d = self.server.守护
        if p == '/status':
            return self._j({'status': 'ok', '已学条数': d.总计数, '分类数': len(d.网络.标签映射), '蒸馏层': d.蒸馏器 is not None, '海马体': d.海马体 is not None, '锚点': d.锚点 is not None, '上下文': d.上下文 is not None, '图谱': d.图谱 is not None})
        elif p == '/predict':
            t = q.get('t', [None])[0] or q.get('text', [None])[0]
            if not t: return self._j({'ok': False}, 400)
            return self._j({'ok': True, '预测': d.预测(t)})
        elif p == '/search':
            qq = q.get('q', [None])[0]
            if not qq: return self._j({'ok': False}, 400)
            return self._j({'ok': True, '结果': d.搜索(qq), '数量': len(d.搜索(qq))})
        return self._j({'ok': False}, 404)
    def do_POST(self):
        p = urlparse(self.path).path.rstrip('/')
        try:
            cl = int(self.headers.get('Content-Length', 0))
            b = json.loads(self.rfile.read(cl)) if cl else {}
        except: b = {}
        if not hasattr(self.server, '守护'): self.server.守护 = 简芯守护(); self.server.守护.扫描记忆()
        d = self.server.守护
        if p == '/feed':
            text = b.get('text', ''); source = b.get('source', '')
            if not text or not source: return self._j({'ok': False}, 400)
            return self._j(d.投喂(text, source))
        elif p == '/predict':
            text = b.get('text', '') or b.get('t', '')
            if not text: return self._j({'ok': False}, 400)
            return self._j({'ok': True, '预测': d.预测(text)})
        return self._j({'ok': False}, 404)

if __name__ == '__main__':
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8500
    srv = HTTPServer(('0.0.0.0', port), Handler)
    print(f'[简芯v2] HTTP -> http://localhost:{port}')
    srv.serve_forever()
