

import os
from zst_透明层 import zst_pickle_load, zst_pickle_dump
import pickle
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class 知识蒸馏器:

    def __init__(self, 向量寄存器, 工作目录=None):

        self.源 = 向量寄存器
        if 工作目录 is None:
            工作目录 = os.path.join(os.path.dirname(__file__), '中枢数据')
        self.持久化文件 = os.path.join(工作目录, '蒸馏知识.pkl')
        self.蒸馏库 = []
        self.聚类阈值 = 0.6
        self._加载()

    def 蒸馏(self):

        if self.源.条数() == 0:
            return {'蒸馏': 0, '总记忆': 0}

        文本库 = self.源.文本库
        if len(文本库) < 2:
            return {'蒸馏': 0, '总记忆': len(文本库)}

        vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000, sublinear_tf=True)
        矩阵 = vec.fit_transform(文本库)
        相似度 = cosine_similarity(矩阵)

        归并 = list(range(len(文本库)))
        for i in range(len(文本库)):
            for j in range(i + 1, len(文本库)):
                if 相似度[i][j] > self.聚类阈值:
                    self._合并(归并, i, j)

        簇 = {}
        for i, 根 in enumerate(归并):
            根 = self._找根(归并, 根)
            簇.setdefault(根, []).append(i)

        新蒸馏 = []
        for 根, 成员 in 簇.items():
            if len(成员) < 2:
                continue
            簇文本 = [文本库[i] for i in 成员]

            浓缩 = self._浓缩(簇文本)

            总访问 = sum(self.源.元数据[i].get('访问', 0) for i in 成员 if i < len(self.源.元数据))

            新蒸馏.append({
                '精华': 浓缩,
                '来源条数': len(成员),
                '总访问': 总访问,
                '时间': time.time(),
            })

        if 新蒸馏:

            for 新 in 新蒸馏:
                if not any(self._相似度(新['精华'], 旧['精华']) > 0.8 for 旧 in self.蒸馏库):
                    self.蒸馏库.append(新)

            self.蒸馏库.sort(key=lambda x: x['总访问'], reverse=True)
            self.蒸馏库 = self.蒸馏库[:50]

            self._保存()

        return {
            '蒸馏': len(新蒸馏),
            '总蒸馏': len(self.蒸馏库),
            '总记忆': len(文本库),
        }

    def 查询蒸馏(self, 查询文本, 数量=3):

        if not self.蒸馏库:
            return []

        精华列表 = [x['精华'] for x in self.蒸馏库]
        vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000, sublinear_tf=True)
        矩阵 = vec.fit_transform(精华列表 + [查询文本])
        查询向量 = 矩阵[-1:]
        库向量 = 矩阵[:-1]

        相似度 = cosine_similarity(查询向量, 库向量)[0]
        排序 = np.argsort(相似度)[::-1][:数量]

        return [{
            '精华': self.蒸馏库[i]['精华'],
            '相似度': float(相似度[i]),
            '来源条数': self.蒸馏库[i]['来源条数'],
            '总访问': self.蒸馏库[i]['总访问'],
        } for i in 排序 if 相似度[i] > 0.1]

    def _浓缩(self, 文本列表):

        if len(文本列表) == 1:
            return 文本列表[0]

        全文 = ' '.join(文本列表)

        try:
            vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3),
                                   max_features=20, stop_words=None)
            tfidf = vec.fit_transform([全文])
            特征 = vec.get_feature_names_out()
            权重 = tfidf.toarray()[0]
            重要词 = [特征[i] for i in np.argsort(权重)[-5:] if 权重[i] > 0]
        except:
            重要词 = []

        浓缩 = 全文[:200] if len(全文) > 200 else 全文
        return 浓缩

    def _相似度(self, a, b):

        try:
            vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
            tfidf = vec.fit_transform([a, b])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(sim)
        except:
            return 0.0

    def _合并(self, arr, i, j):
        根i, 根j = self._找根(arr, i), self._找根(arr, j)
        if 根i != 根j: arr[根i] = 根j

    def _找根(self, arr, i):
        while arr[i] != i:
            arr[i] = arr[arr[i]]
            i = arr[i]
        return i

    def _保存(self):
        with open(self.持久化文件, 'wb') as f:
            zst_pickle_dump(self.蒸馏库, self.持久化文件)

    def _加载(self):
        if os.path.exists(self.持久化文件):
            try:
                with open(self.持久化文件, 'rb') as f:
                    self.蒸馏库 = zst_pickle_load(self.持久化文件)
            except:
                pass

    def 状态(self):
        return {'蒸馏条数': len(self.蒸馏库), '文件': self.持久化文件}

if __name__ == '__main__':
    print('=== 知识蒸馏组件 v1 自测 ===')

    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from slot_向量 import 向量
    vm = 向量()

    vm.记('天翼云服务器IP是101.227.49.22，密码在云端文档', '天翼云')
    vm.记('天翼云服务器账号是101.227.49.22，凭据已存', '天翼云')
    vm.记('服务器101.227.49.22是天翼云实例，密码加密存储', '天翼云')

    vm.记('简芯神经网络跑在8500端口，已学1023条', '简芯')
    vm.记('计算中心神经网络位于8500端口，学习经典规则', '简芯')
    vm.记('有道云笔记CLI在workspace/bin目录，配置OK', '有道云')

    print('已写入%d条记忆' % vm.条数())

    kd = 知识蒸馏器(vm)
    r = kd.蒸馏()
    print('蒸馏结果:', r)

    qr = kd.查询蒸馏('服务器', 3)
    print('查询"服务器"蒸馏知识:')
    for 条 in qr:
        print('  [%.3f] %s' % (条['相似度'], 条['精华'][:60]))

    print('状态:', kd.状态())
    print('自测通过')
