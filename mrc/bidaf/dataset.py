# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter
import itertools



def _padding_from_tokens(list_of_tokens,pad_id,max_len):
    lens = [len(tokens) for tokens in list_of_tokens]
    pad_len = _caculate_maxlen_batch(lens,max_len)
    padded = _padding_batch(list_of_tokens,pad_len,pad_id)
    return  padded,lens

def _caculate_maxlen_batch(actual_lens,defined_max_len):
    return  min(defined_max_len, max(actual_lens))

def _padding_batch(list_of_tokens,pad_len,pad_id):
    return [(ids + [pad_id] * (pad_len - len(ids)))[: pad_len] for ids in list_of_tokens] 

def _dataset_path_meta(path):
    dirpath =os.path.dirname(path)
    base =os.path.basename(path)
    filename_without_ext = os.path.splitext(base)[0]
    return {'filename':filename_without_ext,'block_dir':'%s/%s.blocks'%(dirpath,filename_without_ext)}

def _get_kth_dataset_block_path(block_dir,block_num):
    return '%s/%d.seg'%(block_dir,block_num)


def test_data_fetch():
    tf = SampleTransformer(False,False,5)
    fetcher = DataFetcher('./data/preprocessed/trainset/search.train.30000.json',tf)
    for data in iter(fetcher):
        pass
    it = iter(fetcher)
    l = next_k_items(it,32)
    for x in l:
        print(x['question'])



def test_brcd_dataset(funcname):
    from vocab import Vocab

   
    print('load dataset')
    brc_data = BRCDataset( 5,500, 60, ['./data/preprocessed/trainset/search.train.30000.json'],\
                           ['./data/preprocessed/devset/search.dev.json'], ['./data/demo/testset/search.test.json'])
    vocab = Vocab(lower=True)
    print('build vocab')
    for word in brc_data.word_iter('train'):
        vocab.add(word)
    print('vocab size %d'%(vocab.size()))
    print('test:word end')
    if funcname == 'word':
        return
    
    brc_data.set_vocab(vocab)


    #__batch =  [dict(zip( batch,t)) for t in zip(* batch.values())]
    #print('lenght of batch')
    #print(len(__batch))
    #print(__batch[0])
    for i,batch in enumerate(brc_data.gen_mini_batches('train',32)):
        if i % 50==0:
            print('%dth batch'%(i))
        bb = [dict(zip( batch,t)) for t in zip(* batch.values())]
        #print(bb[0])
        print(len(bb))

    

    
  
        
def split_dataset_to_blocks(dataset_path,block_size=4096):
    meta = _dataset_path_meta(dataset_path)
    if not os.path.exists(meta['block_dir']):
        os.mkdir(meta['block_dir'])

    with open(dataset_path,'r',encoding='utf-8') as f:
        block_cnt = 0
        block = []
        for i,l in enumerate(f):
            if i%1000==0:
                print('preprocessing %dth data'%(i))
            block.append(l)
            if len(block) % block_size == 0:
                wf = open(_get_kth_dataset_block_path(meta['block_dir'],block_cnt),'w',encoding='utf-8')
                for l in block:
                    wf.write(l)
                block = []
                block_cnt+=1

def next_k_items(iterable,k):
    l = []
    cnt = 0
    while cnt<k:
        try:
            item = next(iterable)
        except StopIteration:        
            break
        l.append(item)   
        cnt+=1
    return l


def get_pad_id(vocab):
    return vocab.get_id(vocab.pad_token)


class SampleTransformer():
    def __init__(self,train,gold_paragraph,max_p_len):
        self.train = train
        self.gold_paragraph = gold_paragraph
        self.max_p_len = max_p_len
    
    def check_wellformed_sample(self,sample):
        if self.train:
            if len(sample['answer_spans']) == 0:
                return False
            if sample['answer_spans'][0][1] >= self.max_p_len:
                return False
        if  self.gold_paragraph and  not self.train and ('answer_docs' not in sample or len( sample['answer_docs'])==0):
                return False
        return True

    def process_sample(self,sample):
        sample['question_tokens'] = sample['segmented_question']
        sample['passages'] = []
        if 'answer_docs' in sample:
            sample['answer_passages'] = sample['answer_docs']
            #if gold_paragraph and  not train:
            #    answer_doc = sample['documents'][sample['answer_docs'][0]]
            #    most_rel_para =  answer_doc['most_related_para']
            #    sample['passages'].append({'passage_tokens': answer_doc['segmented_paragraphs'][most_rel_para]})
        for _, doc in enumerate(sample['documents']):
            if self.train:
                most_related_para = doc['most_related_para']
                sample['passages'].append(
                    {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                     'is_selected': doc['is_selected']}
                )
            else:
                if not self.gold_paragraph:
                    para_infos = []
                    for para_tokens in doc['segmented_paragraphs']:
                        question_tokens = sample['segmented_question']
                        common_with_question = Counter(para_tokens) & Counter(question_tokens)
                        correct_preds = sum(common_with_question.values())
                        if correct_preds == 0:
                            recall_wrt_question = 0
                        else:
                            recall_wrt_question = float(correct_preds) / len(question_tokens)
                        para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                    para_infos.sort(key=lambda x: (-x[1], x[2]))
                    fake_passage_tokens = []
                    for para_info in para_infos[:1]:
                        fake_passage_tokens += para_info[0]
                    sample['passages'].append({'passage_tokens': fake_passage_tokens})
                else:
                     most_rel_para =  doc['most_related_para']
                     sample['passages'].append({'passage_tokens': doc['segmented_paragraphs'][most_rel_para]})

class DataFetcher():
    def __init__(self,filepath,transformer):
        self.iter_index = 0

        self.buffer = []
        self.index_in_buffer = 0
        self.current_file_block_num = 0
        self.transformer = transformer

        self.block_mode = True
        self.pathmeta = _dataset_path_meta(filepath)
        print('load %s'%(filepath),flush=True)
        if not os.path.exists(self.pathmeta["block_dir"]):
            print('file mode %s'%(filepath))
            self.block_mode = False
            self._fill_buffer_from_file(filepath)

        

    
    def _fill_buffer_from_file(self,path):
        print('open file %s'%(path))
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                if not self.transformer.check_wellformed_sample(sample):
                    continue
                self.transformer.process_sample(sample)   
                self.buffer.append(sample)   

    def _open_next_file_to_buffer(self):
        self.index_in_buffer = 0
        self.buffer = []
        path = _get_kth_dataset_block_path(self.pathmeta["block_dir"],self.current_file_block_num)
        if os.path.exists(path):
            self._fill_buffer_from_file(path)
            self.current_file_block_num+=1
   

    def _get_nextitem_from_buffer(self):
        item = self.buffer[self.index_in_buffer]
        self.index_in_buffer+=1
        return item


    def __iter__(self):
        if self.block_mode == False:
            return iter(self.buffer)
        self.iter_index = 0
        self.current_file_block_num = 0
        return self

    def __next__(self):
        
        if self.index_in_buffer == len(self.buffer):
           self._open_next_file_to_buffer()
           if len(self.buffer) == 0:
                raise StopIteration
#        if self.iter_index%1000 == 0:
#            print('handle {}th line'.format(self.iter_index))
        self.iter_index+=1
        return self._get_nextitem_from_buffer()





class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[],gold_paragraph=False):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.gold_paragraph = gold_paragraph
        self.vocab  = None


        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set.append(self._load_dataset(train_file, train=True))
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set.append(self._load_dataset(dev_file))
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))
        if test_files:
            for test_file in test_files:
                self.logger.info('load %s'%(test_file))
                self.test_set.append(self._load_dataset(test_file))
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        transformer = SampleTransformer(train,self.gold_paragraph,self.max_p_len)
        dataset = DataFetcher(data_path,transformer)
        return dataset

    def dataset_iterator(self,fetcher_list):
        return itertools.chain.from_iterable([  iter(x) for x in fetcher_list])


    def set_vocab(self,vocab):
        self.vocab = vocab


    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        it = self.dataset_iterator(data_set)
        for i,sample in enumerate(it):
            if i%2000 == 0:
                self.logger.info('worditer {}th sample'.format(i))
            for token in sample['question_tokens']:
                yield token
            for passage in sample['passages']:
                for token in passage['passage_tokens']:
                    yield token


    def numeralize_sample(self,sample):
        sample['question_token_ids'] = self.vocab.convert_to_ids(sample['question_tokens'])
        for passage in sample['passages']:
            passage['passage_token_ids'] = self.vocab.convert_to_ids(passage['passage_tokens'])


    def gen_mini_batches(self, set_name, batch_size, shuffle=False):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        assert self.vocab is not None
        if set_name == 'train':
            dataset = self.train_set
        elif set_name == 'dev':
            dataset = self.dev_set
        elif set_name == 'test':
            dataset = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        #data_size = len(data)
        #indices = np.arange(data_size)
        #if shuffle:
        #    np.random.shuffle(indices)
        it = self.dataset_iterator(dataset)
        while True:
            #batch_indices = indices[batch_start: batch_start + batch_size]
            raw_datas = next_k_items(it,batch_size)
            if  len(raw_datas) == 0:
                break
            for sample in raw_datas:
                self.numeralize_sample(sample)
            batch =  self._one_mini_batch(raw_datas)
            yield batch

    def _one_mini_batch(self, datas):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': datas,
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        return batch_data

    def _dynamic_padding(self, batch_data):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_id = get_pad_id(self.vocab)
        pad_p_len = _caculate_maxlen_batch(batch_data['passage_length'],self.max_p_len)
        pad_q_len = _caculate_maxlen_batch(batch_data['question_length'],self.max_q_len)
        batch_data['passage_token_ids'] =  _padding_batch(batch_data['passage_token_ids'],pad_p_len,pad_id)                           
        batch_data['question_token_ids'] = _padding_batch(batch_data['question_token_ids'],pad_q_len,pad_id)
        return batch_data, pad_p_len, pad_q_len
#split_dataset_to_blocks('../data/preprocessed/trainset/search.train.json')
#split_dataset_to_blocks('../data/preprocessed/trainset/zhidao.train.json')
#split_dataset_to_blocks('../data/preprocessed/devset/search.dev.json')
#split_dataset_to_blocks('../data/preprocessed/devset/zhidao.dev.json')
#split_dataset_to_blocks('./data/preprocessed/trainset/search.train.30000.json')
#split_dataset_to_blocks('../data/preprocessed/testset/search.test.json')
#split_dataset_to_blocks('../data/preprocessed/testset/zhidao.test.json')
#test_data_fetch()
#test_brcd_dataset('batch')
