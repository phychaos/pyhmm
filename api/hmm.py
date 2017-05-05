#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-5-4
    @author: Linlifang
    @file: hmm.py
    @time: 17-5-4.下午6:18
"""
import datetime
import os
import pickle

import copy
import re


class HMM(object):
    def __init__(self, default_prob=0.000000000001, model='model/model'):
        """
        初始化数据
        :param default_prob: 默认概率
        """
        self.default_prob = default_prob
        self.model = model
        self.v = set()  # 观测集合, 字符数
        self.q = set()  # 状态集合, 标签数
        self.a = {}     # 状态转移矩阵
        self.b = {}     # 观测概率矩阵
        self.pi = {}    # 初始状态概率向量

    def train(self, train_path):
        """
        训练模型, 状态转移矩阵, 观测概率矩阵
        :param train_path: 标注数据文件路径
        :return:
        """
        print('开始训练......')
        start = datetime.datetime.now()
        print('开始时间\t', start, '\n......\n')
        # 计算概率转移矩阵, 观测状态矩阵
        tag_data = self.read_data(train_path)
        for sentence in tag_data:
            pre_s = -1
            for o, s in sentence:
                self.b[s][o] = self.b.setdefault(s, {}).setdefault(o, 0) + 1
                if pre_s == -1:
                    self.pi[s] = self.pi.setdefault(s, 0) + 1
                else:
                    self.a[pre_s][s] = self.a.setdefault(pre_s, {}).setdefault(s, 0) + 1
                pre_s = s

        # 概率归一化, a:状态转移矩阵Σj(a[i][j])=1, 观测概率Σj(b[i][j])=1
        for key, value in self.a.items():
            prob_sum = 0
            for k, v in value.items():
                prob_sum += v
            for k, v in value.items():
                self.a[key][k] = 1.0 * v / prob_sum

        for key, value in self.b.items():
            prob_sum = 0
            for k, v in value.items():
                prob_sum += v
            for k, v in value.items():
                self.b[key][k] = 1.0 * v / prob_sum
        # 初始状态
        prob_sum = sum(self.pi.values())
        for k, v in self.pi.items():
            self.pi[k] = 1.0 * v / prob_sum

        end = datetime.datetime.now()
        print('训练完成.....')
        print('结束时间\t', end)
        print('\n训练耗时\t', end - start)
        self.save_model()

    def test(self, data):
        """
        测试数据, 分词
        :param data: 字符串
        :return: 分词结果, 列表
        """
        p = re.compile('BM*E|S')
        print('开始测试...')
        self.get_model()
        predict = self.predict(data)
        result = []
        for i in p.finditer(''.join(predict)):
            start, end = i.span()
            word = data[start:end]
            result.append(word)
        return result

    def predict(self, data):
        """
        预测序列状态
        :param data: 字符串
        :return: 状态
        """
        prob = self.default_prob
        min_prob = -1 * float('inf')
        w = [{} for _ in data]
        path = {}
        for s in self.q:
            w[0][s] = 1.0 * self.pi.get(s, prob) * self.b.get(s, {}).get(data[0], prob)
            path[s] = [s]
        for t in range(1, len(data)):
            new_path = {}
            for s in self.q:
                max_prob = min_prob
                max_s = ''
                for pre_s in self.q:
                    probs = w[t - 1][pre_s] * self.a.get(pre_s, {}).get(s, prob) * self.b.get(s, {}).get(data[t], prob)
                    max_prob, max_s = max((max_prob, max_s), (probs, pre_s))
                w[t][s] = max_prob
                tmp = copy.deepcopy(path[max_s])
                tmp.append(s)
                new_path[s] = tmp
            path = new_path
        max_prob, max_s = max((w[len(data) - 1][s], s) for s in self.q)
        return path[max_s]

    def get_model(self):
        """
        加载模型
        :return:
        """
        print('加载模型...')
        model_name = os.path.join(os.getcwd(), self.model)
        with open(model_name, 'rb') as fp:
            model = pickle.load(fp)
        self.a, self.b, self.pi, self.v, self.q = model

    def save_model(self):
        """
        保存模型
        :return:
        """
        print('\n保存模型...')
        model = [self.a, self.b, self.pi, self.v, self.q]
        model_name = os.path.join(os.getcwd(), self.model)
        with open(model_name, 'wb') as fp:
            pickle.dump(model, fp)

    def read_data(self, filename):
        """
        读取已标注数据
        :param filename: 标注数据路径
        :return:
        """
        tag_data = []
        sentence = []
        with open(filename, 'r') as fp:
            for raw in fp.readlines():
                line = raw.strip().split('\t')
                if len(line) == 2:
                    self.v.add(line[0])
                    self.q.add(line[1])
                    sentence.append(line)
                else:
                    tag_data.append(sentence) if sentence else None
                    sentence = []
        return tag_data
