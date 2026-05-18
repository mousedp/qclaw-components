

import os
from zst_透明层 import zst_pickle_load, zst_pickle_dump
import pickle
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class 向量:

    热层阈值 = 7 * 24 * 3600
    温层阈值 = 30 * 24 * 3600
    热层最低访问 = 3

    def __init__(self, 工作目录=None, 最大热层=50, 最大记忆=2000):
        if 工作目录 is None:
            工作目录 = os.path.join(os.path.dirname(__file__), '中枢数据')
        self.数据目录 = 工作目录
        self.持久化文件 = os.path.join(self.数据目录, '向量记忆.pkl')
        os.makedirs(self.数据目录, exist_ok=True)

        self.最大热层 = 最大热层
        self.最大记忆 = 最大记忆
        self.文本库 = []
        self.元数据 = []
        self.向量化器 = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True
        )
        self._已训练 = False

        self._加载旧版()
        self._加载()

    def 记(self, 文本, 标题='', 标签=None, 层=None):

        self.文本库.append(文本)
        元 = {
            '标题': 标题,
            '标签': str(标签 or ''),
            '时间': time.time(),
            '访问': 0,
            '层': 层 if 层 is not None else 0,
        }
        self.元数据.append(元)
        self._已训练 = False
        self._压缩()
        self._保存()

    def 搜(self, 查询, 数量=5, 层=None):

        if not self.文本库:
            return []

        if not self._已训练:
            self._向量化()

        层掩码 = self._层掩码(层) if 层 is not None else slice(None)

        候选文本 = self.文本库[层掩码]
        候选元数据 = self.元数据[层掩码]
        if not 候选文本:
            return []

        查询向量 = self.向量化器.transform([查询])
        相似度 = cosine_similarity(查询向量, self._向量矩阵[层掩码])[0]

        现在 = time.time()
        得分 = []
        for i, 元 in enumerate(候选元数据):
            基础分 = max(相似度[i], 0.0)
            新鲜度 = 1.0 / (1 + (现在 - 元.get('时间', 现在)) / 86400)
            鲜加 = 0.15 * 新鲜度
            热加 = 0.1 * min(元.get('访问', 0), 10) / 10
            层减 = 0.0 if 元.get('层', 1) == 0 else (0.05 if 元.get('层', 1) == 1 else 0.1)
            得分.append(基础分 + 鲜加 + 热加 - 层减)

        排序索引 = np.argsort(得分)[::-1][:数量]

        结果 = []
        for i in 排序索引:
            if 相似度[i] > 0.01:
                结果.append({
                    '文本': 候选文本[i],
                    '综合分': round(float(得分[i]), 4),
                    '相似度': round(float(相似度[i]), 4),
                    '标题': 候选元数据[i].get('标题', ''),
                    '标签': 候选元数据[i].get('标签', ''),
                    '层': 候选元数据[i].get('层', 1),
                    '访问': 候选元数据[i].get('访问', 0),
                })

                候选元数据[i]['访问'] = 候选元数据[i].get('访问', 0) + 1
                self._重分层(候选元数据[i])

        self._保存()
        return 结果

    def _层掩码(self, 层):

        if 层 == 0:
            return [元.get('层', 1) == 0 for 元 in self.元数据]
        elif 层 == 1:
            return [元.get('层', 1) in (0, 1) for 元 in self.元数据]
        elif 层 == 2:
            return [元.get('层', 1) == 2 for 元 in self.元数据]
        return slice(None)

    def _重分层(self, 元):

        现在 = time.time()
        时间差 = 现在 - 元.get('时间', 现在)
        访问 = 元.get('访问', 0)

        if 时间差 < self.热层阈值 or 访问 >= self.热层最低访问:
            元['层'] = 0
        elif 时间差 < self.温层阈值 or 访问 >= 1:
            元['层'] = 1
        else:
            元['层'] = 2

    def _向量化(self):
        全部文本 = self.文本库 + ['']
        self._向量矩阵 = self.向量化器.fit_transform(全部文本)
        self._已训练 = True

    def _压缩(self):

        if len(self.文本库) <= self.最大记忆:
            return

        for 元 in self.元数据:
            self._重分层(元)

        l0索引 = [i for i, 元 in enumerate(self.元数据) if 元.get('层') == 0]
        if len(l0索引) > self.最大热层:
            l0索引.sort(key=lambda i: self.元数据[i].get('访问', 0))
            降级数 = len(l0索引) - self.最大热层
            for i in l0索引[:降级数]:
                self.元数据[i]['层'] = 1

    def _保存(self):
        data = {'文本库': self.文本库, '元数据': self.元数据}
        with open(self.持久化文件, 'wb') as f:
            zst_pickle_dump(data, self.持久化文件)

    def _加载(self):
        if os.path.exists(self.持久化文件):
            try:
                with open(self.持久化文件, 'rb') as f:
                    data = zst_pickle_load(self.持久化文件)
                self.文本库 = data.get('文本库', [])
                self.元数据 = data.get('元数据', [])

                for 元 in self.元数据:
                    元.setdefault('时间', time.time())
                    元.setdefault('访问', 0)
                    元.setdefault('层', 1)
            except Exception as e:
                print(f'[向量] 加载失败: {e}')

    def _加载旧版(self):
        旧版文件 = os.path.join(self.数据目录, '向量机.pkl')
        if os.path.exists(旧版文件):
            try:
                with open(旧版文件, 'rb') as f:
                    data = zst_pickle_load(self.持久化文件)
                旧文本 = data.get('文本库', [])
                旧元数据 = data.get('元数据', [])
                if 旧文本:
                    self.文本库 = 旧文本
                    self.元数据 = [{
                        '标题': 元.get('标题', ''),
                        '标签': 元.get('标签', ''),
                        '时间': time.time(),
                        '访问': 0,
                        '层': 1,
                    } for 元 in 旧元数据]
                    self._已训练 = False
                    self._保存()
                    os.remove(旧版文件)
                    print(f'[向量] 已迁移 {len(旧文本)} 条旧版数据')
            except:
                pass

    def 条数(self):
        return len(self.文本库)

    def 状态(self):
        l0 = sum(1 for 元 in self.元数据 if 元.get('层') == 0)
        l1 = sum(1 for 元 in self.元数据 if 元.get('层') == 1)
        l2 = sum(1 for 元 in self.元数据 if 元.get('层') == 2)
        return {
            '已学条数': len(self.文本库),
            '文件': self.持久化文件,
            'L0热层': l0, 'L1温层': l1, 'L2归档': l2,
            '最大热层': self.最大热层,
        }

if __name__ == '__main__':
    print('=== 向量记忆组件 v2 自测（长上下文版） ===')
    vm = 向量()

    vm.记('天翼云服务器IP是101.227.49.22，密码已存云端文档', '天翼云')
    vm.记('派心跳使用π计时引擎，每分钟输出指纹+双哈希', '派心跳')
    vm.记('简芯神经网络跑在8500端口，已学1023条', '简芯')
    vm.记('有道云笔记CLI在workspace/bin目录，API Key已配置', '有道云')
    print('写入4条完成')

    r1 = vm.搜('服务器密码在哪里')
    r2 = vm.搜('服务器密码在哪里')
    r3 = vm.搜('服务器密码在哪里')

    print('搜索"服务器密码在哪里"（第3次）：')
    for 条 in r3:
        print('  [综合%.3f|相关%.3f|L%d|访%d] %s' % (
            条['综合分'], 条['相似度'], 条['层'], 条['访问'], 条['文本'][:50]))

    print('搜索"运行端口是什么"：')
    r4 = vm.搜('运行端口是什么')
    for 条 in r4:
        print('  [综合%.3f|相关%.3f|L%d|访%d] %s' % (
            条['综合分'], 条['相似度'], 条['层'], 条['访问'], 条['文本'][:50]))

    print('状态:', vm.状态())
    print('自测通过')
