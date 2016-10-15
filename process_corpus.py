#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_corpus_1 is used to process corpus where there are contents after a colon and change them to one word every line
Example of one line in raw corpus:  环境科学:环境科学,环科,环境科学,环境
after processed: 环境科学
                 环科
                 环境科学
                 环境

process_corpus_2 is used to combine all the dictionaries into one
"""

import os
import optparse


class Process_Corpus:

    def __init__(self, dict_dir):
        self.dict_dir = dict_dir

    def process_corpus_1(self):
        f1 = open('dict_edu1.txt', 'w')
        fname_list = os.listdir(self.dict_dir)
        for fname in fname_list:  # for every dictionary file in the directory
            with open(self.dict_dir + '//' + fname, 'r')as f:
                f2 = f.readlines()
            for line in f2:  # for every line in the dictionary file
                list=line.split(":")[1].split(",")  # split the line into 2 parts by colon and split 2nd part by comma
                for w in list:
                    w = w.strip()
                    f1.write(w)   
                    f1.write('\n')

    def process_corpus_2(self):
        f1 = open('dict_edu.txt', 'w')
        fname_list = os.listdir(self.dict_dir)
        for fname in fname_list:
            with open(self.dict_dir + '//' + fname, 'r')as f:
                f2 = f.readlines()
            for w in f2:
                w = w.strip()
                f1.write(w)
                f1.write('\n')

if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("-d", dest="dict_dir", help="directory of dictionary")
    parser.add_option("-t", dest="type", help="1 or 2")
    (options, args) = parser.parse_args()

    if options.type == '1':
        pc = Process_Corpus(options.dict_dir)
        pc.process_corpus_1()
    if options.type == '2':
        pc = Process_Corpus(options.dict_dir)
        pc.process_corpus_2()