#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-5-4
    @author: Linlifang
    @file: main.py
    @time: 17-5-4.下午4:42
"""
from api.hmm import HMM
from config.config import *


def train():
    hmm = HMM(model='model/model')
    hmm.train(tag_data_path)


def test():
    a = '要继续加强追逃追赃等务实合作以及社会领域交流互鉴，深化旅游、大熊猫合作研究和足球等人文领域交流合作。'
    hmm = HMM()
    result = hmm.test(a)
    print(result)


if __name__ == '__main__':
    # train()
    test()
