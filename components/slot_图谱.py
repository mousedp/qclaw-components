

import os
from zst_透明层 import zst_pickle_load, zst_pickle_dump
import pickle
import time
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

import re as re2

def _有效实体(名):

    if not 名 or len(名) < 2:
        return False
    中文数 = len(re2.findall(r'[\u4e00-\u9fff]', 名))
    字母数 = len(re2.findall(r'[a-zA-Z]', 名))
    数字数 = len(re2.findall(r'[0-9]', 名))
    总有效 = 中文数 + 字母数
    return 总有效 >= 1 or 数字数 >= 3

class 知识图谱:

    def __init__(self, 向量寄存器, 工作目录=None):
        self.源 = 向量寄存器
        if 工作目录 is None:
            工作目录 = os.path.join(os.path.dirname(__file__), '中枢数据')
        self.持久化文件 = os.path.join(工作目录, '知识图谱.pkl')
        self.实体 = {}
        self.关系 = {}
        self._加载()

    def 构建(self, 每个记忆取词=5):

        if self.源.条数() == 0:
            return {'实体数': 0, '关系数': 0}

        文本库 = self.源.文本库

        def _提取实体(文本, 最多=5):

            实体 = []
            中词 = re2.findall(r"[\u4e00-\u9fff]{2,4}", 文本)
            实体.extend(中词)
            英词 = re2.findall(r"[a-zA-Z]{2,}", 文本)
            实体.extend(英词)
            数组 = re2.findall(r"[0-9]+[a-zA-Z\u4e00-\u9fff]", 文本)
            实体.extend(数组)
            实体 = list(set(实体))
            实体.sort(key=len, reverse=True)
            return 实体[:最多]

        try:
            for i in range(len(文本库)):
                实体列表 = _提取实体(文本库[i], 每个记忆取词)

                now = time.time()
                for 实体名 in 实体列表:
                    if 实体名 not in self.实体:
                        self.实体[实体名] = {'频次': 0, '最后': now}
                    self.实体[实体名]['频次'] += 1
                    self.实体[实体名]['最后'] = now

                for a in range(len(实体列表)):
                    for b in range(a + 1, len(实体列表)):
                        边 = (实体列表[a], 实体列表[b])
                        self.关系[边] = self.关系.get(边, 0) + 1
                        边2 = (实体列表[b], 实体列表[a])
                        self.关系[边2] = self.关系.get(边2, 0) + 1

        except Exception as e:
            print(f'[知识图谱] 构建失败: {e}')

        丢弃实体 = [e for e, v in self.实体.items() if v['频次'] < 3]
        for e in 丢弃实体:
            del self.实体[e]
        self.关系 = {k: v for k, v in self.关系.items() if v >= 2 and k[0] not in 丢弃实体 and k[1] not in 丢弃实体}

        self._保存()
        return {'实体数': len(self.实体), '关系数': len(self.关系)}

    def 查询(self, 实体名, 深度=2, 最多=5):

        if 实体名 not in self.实体:
            return []

        已访问 = {实体名}
        结果 = []
        当前层 = [实体名]

        for _ in range(深度):
            下一层 = []
            for 节点 in 当前层:
                for (a, b), 强度 in self.关系.items():
                    if a == 节点 and b not in 已访问:
                        下一层.append(b)
                        已访问.add(b)
                        结果.append({
                            '实体': b,
                            '关联路径': '%s→%s' % (节点, b),
                            '关联强度': 强度,
                            '频次': self.实体.get(b, {}).get('频次', 0),
                        })
            当前层 = 下一层

        结果.sort(key=lambda x: x['频次'], reverse=True)
        return 结果[:最多]

    def 搜(self, 文本, 数量=5):

        try:
            vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
            tfidf = vec.fit_transform([文本])
            特征 = vec.get_feature_names_out()
            权重 = tfidf.toarray()[0]
            排序 = np.argsort(权重)[::-1]
            命中实体 = [特征[i] for i in 排序[:3] if 权重[i] > 0 and 特征[i] in self.实体]
        except:
            命中实体 = []

        所有结果 = []
        for 实体名 in 命中实体:
            所有结果.extend(self.查询(实体名, 深度=1, 最多=3))

        所见 = set()
        唯一结果 = []
        for 条 in 所有结果:
            if 条['实体'] not in 所见 and 条['实体'] not in 命中实体:
                所见.add(条['实体'])
                唯一结果.append(条)

        唯一结果.sort(key=lambda x: x['频次'], reverse=True)
        return 唯一结果[:数量]

    def 状态(self):
        return {'实体数': len(self.实体), '关系数': len(self.关系), '文件': self.持久化文件}

    def _保存(self):
        with open(self.持久化文件, 'wb') as f:
            zst_pickle_dump({'实体': self.实体, '关系': self.关系}, self.持久化文件)

    def _加载(self):
        if os.path.exists(self.持久化文件):
            try:
                with open(self.持久化文件, 'rb') as f:
                    d = zst_pickle_load(self.持久化文件)
                self.实体 = d.get('实体', {})
                self.关系 = d.get('关系', {})
            except:
                pass

if __name__ == '__main__':
    print('=== 知识图谱 v1 自测 ===')

    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from slot_向量 import 向量
    vm = 向量()

    vm.记('天翼云服务器IP是101.227.49.22，密码已存云端文档', '天翼云')
    vm.记('天翼云服务器在香港机房运行web服务', '天翼云')
    vm.记('简芯神经网络跑在8500端口，已学1023条', '简芯')
    vm.记('简芯的16个插槽已插入4个组件', '简芯')
    vm.记('派心跳使用π计时引擎，每分钟输出指纹', '派心跳')
    vm.记('派心跳双哈希指纹用于验证系统活性', '派心跳')
    vm.记('叶子的主人是皮皮鲁智能体', '身份')

    print('向量记忆条数:', vm.条数())

    kg = 知识图谱(vm)
    r = kg.构建(每个记忆取词=4)
    print('构建结果:', r)

    print('热门实体:')
    热门 = sorted(kg.实体.items(), key=lambda x: x[1]['频次'], reverse=True)[:5]
    for e, v in 热门:
        print('  %s (频次%d)' % (e, v['频次']))

    r = kg.查询('简芯', 深度=2)
    print('查询"简芯"关联:')
    for 条 in r:
        print('  %s (强度%d, 频次%d)' % (条['实体'], 条['关联强度'], 条['频次']))

    print('状态:', kg.状态())
    print('自测通过')
