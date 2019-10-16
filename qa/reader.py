from common.util import get_default_device,load_json_config,Factory
import torch
import itertools
from bert.tokenization import BertTokenizer
from mrc.bert.util import  load_bert_rc_model,BertInputConverter
from dataloader.dureader import BertRCDataset
from .decoder import MrcDecoderFactory
import pandas as pd
import numpy as np
#import tensorflow as tf
import os
import pickle


# convert rawfields to dict
def torchtext_batch_to_dictlist(batch):
    d = {}
    fields_names = list(batch.fields)
    d = { k:getattr(batch,k) for k in fields_names if not isinstance(getattr(batch,k),torch.Tensor)}
    l = pd.DataFrame(d).to_dict('records')
    return l




# start_probs/end_probs list :[prob1,prob2....]
def extract_answer_dp_linear(start_probs,end_probs):
    # max_start_pos[i] max_start_pos end in i to get max score
    N = len(start_probs)
    assert N>0
    max_start_pos = [0 for _ in range(N)]
    for i in range(1,N):
        prob1 = start_probs[max_start_pos[i-1]]
        prob2 = start_probs[i]
        if prob1 >= prob2:
            max_start_pos[i] = max_start_pos[i-1]
        else:
            max_start_pos[i] = i
    max_span = None
    max_score = -100000
    for i in range(N):
        score = start_probs[max_start_pos[i]]+end_probs[i]
        if score > max_score:
            max_span = (max_start_pos[i],i)
            max_score = score
    return  max_span,max_score


# start_probs/end_probs list :[prob1,prob2....]
def extract_answer_brute_force(start_probs,end_probs,k=1):
    passage_len = len(start_probs)
    best_start, best_end, max_prob = -1, -1, 0
    l = []
    for start_idx in range(passage_len):
        for ans_len in range(passage_len):
            end_idx = start_idx + ans_len
            if end_idx >= passage_len:
                continue
            prob = start_probs[start_idx]+end_probs[end_idx]
            l.append((start_idx,end_idx,prob))
            l = list(sorted(l,key=lambda x:x[2],reverse=True))[0:k]
    return  list(map(lambda x:(x[0],x[1]),l)), list(map(lambda x:x[2],l))


class BertReader():
    def __init__(self,config,decoder_dict=None,device=None):
        self.config = config
        if device is None:
            self.device = get_default_device()
        bert_config_path = '%s/bert_config.json'%(config.BERT_SERIALIZATION_DIR)
        self.model = load_bert_rc_model( bert_config_path,config.MODEL_PATH,self.device)
        self.model.load_state_dict(torch.load(config.MODEL_PATH,map_location=self.device))
        self.model = self.model.to( self.device)
        self.model.eval()
        #bert-base-chinese
        self.tokenizer =  BertTokenizer('%s/vocab.txt'%(config.BERT_SERIALIZATION_DIR), do_lower_case=True)
        if decoder_dict is None:
            self.decoder = MrcDecoderFactory.from_dict({'class':'default','kwargs':{}})
        else:
            self.decoder =  MrcDecoderFactory.from_dict(decoder_dict)
    # documents {'question':[{'passage':...,}]}
    def extract_answer(self,documents,batch_size=16):
        examples = []
        for question,passage_dict_list  in documents.items():
            for dct in passage_dict_list:
                examples.append()
                passage = dct['passage']
                examples.append({'question':question,'passage':passage})
        
        dataset  = BertRCDataset(examples,self.config.max_query_length,self.config.max_seq_length,mode='eval',device=self.device)
        iterator = dataset.make_batchiter(batch_size=batch_size)
        _preds = self.evaluate_on_batch(iterator)
        #for dct in _preds:
        #    question = dct['question']
        #    passage = dct['passage']
        #    passage_dct_l = documents[question]
        #    for d   in passage_dct_l:
        #        if passage == d['passage']:
        #            d.update({'span':dct['span'],'span_score':dct['span_score']})
        return _preds

    # record : list of dict  [ {field1:value1,field2:value2...}}]
    def evaluate_on_records(self,records):
        pass


    def evaluate_on_batch(self,iterator):
        with torch.no_grad():
            preds = []
            for  i,batch in enumerate(iterator):
                if i % 20 == 0:
                    print('evaluate on %d batch'%(i))
                start_probs, end_probs = self.model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask)
                batch_dct_list =  torchtext_batch_to_dictlist(batch)
                for j in range(len(start_probs)):
                    sb,eb = start_probs[j], end_probs[j]
                    sb ,eb  = sb.cpu().numpy(),eb.cpu().numpy()
                    text  = "$" + batch.question[j] + "\n" + batch.passage[j]
                    answer,score,_ = self.decoder.decode(sb,eb,text)
                    batch_dct_list[j].update({'span':answer,'span_score':score})
                    preds.append(batch_dct_list[j])
        return  preds

    def find_best_span_from_probs(self,start_probs, end_probs,policy):
        def greedy():
            best_start, best_end, max_prob = -1, -1, 0
            prob_start, best_start = torch.max(start_probs, 1)
            prob_end, best_end = torch.max(end_probs, 1)
            num = 0
            while True:
                if num > 3:
                    break
                if best_end >= best_start:
                    break
                else:
                    start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0 #寫得很髒....
                    prob_start, best_start = torch.max(start_probs, 1)
                    prob_end, best_end = torch.max(end_probs, 1)
                num += 1
            max_prob = prob_start * prob_end
            if best_start <= best_end:
                return (best_start, best_end), max_prob
            else:
                return (best_end, best_start), max_prob
        return extract_answer_dp_linear(start_probs,end_probs)
    def extact_answer_from_span(self,q,p,span):
        text = "$" + q + "\n" + p
        answer = text[span[0]:span[1]+1]
        return answer





class ReaderFactory(Factory):
    NAME2CLS = {'bert_reader':BertReader,'bidaf':None}
    def __init__(self):
        pass


