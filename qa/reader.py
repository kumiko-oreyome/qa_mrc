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
    def __init__(self,config,decoder_dict=None,eval_flag=True,device=None):
        self.config = config
        if device is None:
            self.device = get_default_device()
        bert_config_path = '%s/bert_config.json'%(config.BERT_SERIALIZATION_DIR)
        self.model = load_bert_rc_model( bert_config_path,config.MODEL_PATH,self.device)
        self.model.load_state_dict(torch.load(config.MODEL_PATH,map_location=self.device))
        self.model = self.model.to( self.device)
        if eval_flag:
            self.model.eval()
        #bert-base-chinese
        self.tokenizer =  BertTokenizer('%s/vocab.txt'%(config.BERT_SERIALIZATION_DIR), do_lower_case=True)
        if decoder_dict is None:
            self.decoder = MrcDecoderFactory.from_dict({'class':'default','kwargs':{}})
        else:
            self.decoder =  MrcDecoderFactory.from_dict(decoder_dict)


    # record : list of dict  [ {field1:value1,field2:value2...}}]
    def evaluate_on_records(self,records,batch_size=64):
        iterator = self.get_batchiter(records,batch_size=batch_size)
        return  self.evaluate_on_batch(iterator)


    def get_batchiter(self,records,train_flag=False,batch_size=64):
        dataset  = BertRCDataset(records,self.config.MAX_QUERY_LEN,self.config.MAX_SEQ_LEN,train_flag=train_flag,device=self.device)
        iterator = dataset.make_batchiter(batch_size=batch_size)
        return iterator


    def evaluate_on_batch(self,iterator):
        preds = []
        with torch.no_grad():
            for  i,batch in enumerate(iterator):
                if i % 20 == 0:
                    print('evaluate on %d batch'%(i))
                preds.extend(self.predict_one_batch(batch))
        return  preds

    def predict_one_batch(self,batch):
        start_probs, end_probs = self.model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask)
        return self.decode_batch(start_probs, end_probs,batch)

    def decode_batch(self,start_probs,end_probs,batch):
        batch_dct_list =  torchtext_batch_to_dictlist(batch)
        preds = []
        for j in range(len(start_probs)):
            sb,eb = start_probs[j], end_probs[j]
            sb ,eb  = sb.cpu().numpy(),eb.cpu().numpy()
            text = "$" + batch.question[j] + "\n" + batch.passage[j]
            answer,score,_ = self.decoder.decode(sb,eb,text)
            #score = score.item() #輸出的score不是機率 所以不會介於0~1之間
            batch_dct_list[j].update({'span':answer,'span_score':score})
            preds.append(batch_dct_list[j]) 
        return preds





class ReaderFactory(Factory):
    NAME2CLS = {'bert_reader':BertReader,'bidaf':None}
    def __init__(self):
        pass


