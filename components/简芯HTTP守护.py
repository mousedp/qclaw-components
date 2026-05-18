import sys, os, json, time, threading

from http.server import HTTPServer, BaseHTTPRequestHandler

from urllib.parse import urlparse, parse_qs



_当前目录 = os.path.dirname(os.path.abspath(__file__))

_工作区 = os.path.abspath(os.path.join(_当前目录, '..', '..'))

sys.path.insert(0, os.path.join(_工作区, 'components', '02-神经网络四合一'))

sys.path.insert(0, os.path.join(_工作区, 'neural-skeleton'))

sys.path.insert(0, os.path.join(_工作区, 'components', '01-记忆组件三合一'))



from 简芯_v2 import 简芯守护

try:

    from memory_component import MemoryCard

except:

    MemoryCard = None



中枢数据 = os.path.join(_当前目录, '中枢数据')

消息目录 = os.path.join(中枢数据, 'messages')

群聊目录 = os.path.join(中枢数据, '群聊')

os.makedirs(消息目录, exist_ok=True)

os.makedirs(群聊目录, exist_ok=True)



记忆卡 = None

if MemoryCard is not None:

    try:

        记忆卡 = MemoryCard(os.path.join(中枢数据, '记忆卡.json'), os.path.join(中枢数据, '记忆保险库'))

        try: 记忆卡.加载(os.path.join(中枢数据, '记忆卡.json'))

        except: 记忆卡 = None

    except: 记忆卡 = None



守护 = 简芯守护()

守护.扫描记忆()

守护.持久化()



def _定时训练():

    while True:

        time.sleep(300)

        try: 守护.扫描记忆()

        except: pass



def _获取时间戳():

    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))



class 简芯Handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):

        self.send_response(200)

        self.send_header('Access-Control-Allow-Origin', '*')

        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

        self.end_headers()



    def _json(self, data, status=200):

        self.send_response(status)

        self.send_header('Content-Type', 'application/json; charset=utf-8')

        self.send_header('Access-Control-Allow-Origin', '*')

        self.end_headers()

        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))



    def _text(self, text, status=200):

        self.send_response(status)

        self.send_header('Content-Type', 'text/plain; charset=utf-8')

        self.send_header('Access-Control-Allow-Origin', '*')

        self.end_headers()

        self.wfile.write(text.encode('utf-8'))



    def _读消息文件(self, 目录):

        消息 = []

        if not os.path.isdir(目录): return 消息

        for fname in sorted(os.listdir(目录), reverse=True)[:100]:

            if not fname.endswith('.json'): continue

            try:

                with open(os.path.join(目录, fname), 'r', encoding='utf-8') as f:

                    消息.append(json.load(f))

            except: pass

        return 消息



    def do_GET(self):

        parsed = urlparse(self.path)

        path = parsed.path.rstrip('/')

        qs = {k: v for k, v in parse_qs(parsed.query).items()}



        if path == '/status':

            标签数 = len(守护.网络.标签映射) if hasattr(守护, '网络') and hasattr(守护.网络, '标签映射') else 0

            return self._json({

                'status': 'ok',

                '已学条数': getattr(守护, '总计数', 0),

                '分类数': 标签数,

                '记忆卡': 记忆卡 is not None,

                '收件箱条数': len(os.listdir(群聊目录)) if os.path.isdir(群聊目录) else 0

            })



        elif path == '/predict':

            t = qs.get('t', [None])[0] or qs.get('text', [None])[0]

            if not t: return self._json({'ok': False, 'error': '需要参数 t 或 text'}, 400)

            result = 守护.预测(t)

            return self._json({'ok': True, '预测': result})



        elif path == '/search':

            q = qs.get('q', [None])[0]

            if not q: return self._json({'ok': False, 'error': '需要参数 q'}, 400)

            return self._json({'ok': True, '结果': 守护.搜索(q)})



        elif path == '/inbox':

            who = qs.get('who', [None])[0]

            if who:

                模式 = '私信'

                目录 = os.path.join(消息目录, who)

            else:

                模式 = '群聊'

                目录 = 群聊目录

            消息列表 = self._读消息文件(目录)

            return self._json({'ok': True, '消息': 消息列表, '总条数': len(消息列表), '模式': 模式})



        elif path == '/clear':

            who = qs.get('who', [None])[0]

            if who:

                目录 = os.path.join(消息目录, who)

            else:

                目录 = 群聊目录

            if os.path.isdir(目录):

                for f in os.listdir(目录):

                    os.remove(os.path.join(目录, f))

            return self._json({'ok': True})



        elif path == '/':

            return self._text('简芯 HTTP 守护\n'

                'GET /status  GET /predict?t=xxx  GET /search?q=xxx\n'

                'GET /inbox(群聊)  GET /inbox?who=xxx(私信)\n'

                'POST /send(群聊)  POST /send to=xxx(私信)\n'

                'POST /feed  POST /predict  GET /clear(群聊)/?who=xxx(私信)')



        return self._json({'ok': False, 'error': '未知路径'}, 404)



    def do_POST(self):

        parsed = urlparse(self.path)

        path = parsed.path.rstrip('/')

        try:

            clen = int(self.headers.get('Content-Length', 0))

            body = json.loads(self.rfile.read(clen).decode('utf-8')) if clen else {}

        except:

            body = {}



        if path == '/predict':

            text = body.get('text', '') or body.get('t', '')

            if not text: return self._json({'ok': False, 'error': '需要 text 字段'}, 400)

            return self._json({'ok': True, '预测': 守护.预测(text)})



        elif path == '/feed':
            text = body.get('text', '')
            source = body.get('source', '')
            if not text or not source: return self._json({'ok': False, 'error': '需要 text 和 source'}, 400)
            result = 守护.投喂(text, source)
            return self._json({'ok': True, **result})



        elif path == '/send':

            from_ = body.get('from', '')

            msg = body.get('msg', '')

            to = body.get('to', None)

            if not from_ or not msg: return self._json({'ok': False, 'error': '需要 from 和 msg'}, 400)



            msg_id = str(int(time.time() * 1000000))



            if to:

                # 私信:保存到 messages/to/

                模式 = '私信'

                目录 = os.path.join(消息目录, to)

            else:

                # 群聊:保存到 群聊目录/

                模式 = '群聊'

                目录 = 群聊目录



            os.makedirs(目录, exist_ok=True)

            条目 = {'id': msg_id, 'from': from_, 'msg': msg, 'time': _获取时间戳()}

            with open(os.path.join(目录, f'{msg_id}.json'), 'w', encoding='utf-8') as f:

                json.dump(条目, f, ensure_ascii=False)



            return self._json({'ok': True, 'id': msg_id, '收件人': to, '已发送': True, '模式': 模式})



        elif path == '/register' or path == '/train':

            守护.扫描记忆()

            return self._json({'ok': True, '已训练': True})



        return self._json({'ok': False, 'error': '未知路径'}, 404)



if __name__ == '__main__':

    PORT = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8500

    线程 = threading.Thread(target=_定时训练, daemon=True)

    线程.start()

    print(f'[简芯] HTTP 守护已启动 (PID={os.getpid()}) -> http://localhost:{PORT}')

    print(f'[简芯] 状态: GET /status  预测: GET /predict?t=xxx')

    srv = HTTPServer(('0.0.0.0', PORT), 简芯Handler)

    srv.serve_forever()

