from types import SimpleNamespace
import io
import json
import torch
import pandas as pd
import mrc.bert.metric.mrc_eval
from .dureader_eval  import  compute_bleu_rouge,normalize
# evaluate by the method of dureader bert probject




# evaluate by the method of bidaf project provided by baidu
def evaluate_mrc_bidaf(pred_answers):
    pred_for_bidaf_eval = {}
    ref_dict = {}
    for qid,v in pred_answers.items():
        best_pred = v[0]
        if len(best_pred['answers']) == 0:
            continue
        pred_for_bidaf_eval[qid] = normalize([ best_pred['span']])
        ref_dict[qid]  = normalize(best_pred['answers'])
    result = compute_bleu_rouge(pred_for_bidaf_eval,ref_dict)
    print(result)
    return result
       

def evaluate_mrc_bert(pred_answers):
    pred_dict_for_eval = {}
    ref_dict_for_eval  = {}
    for _,v in pred_answers.items():
        top1_item = v[0]
        pred_dict_for_eval[top1_item['question_id']] = {'question':top1_item['question'],'question_type': top1_item['question_type'],\
            'answers': [top1_item['span']],'entity_answers': [[]],'yesno_answers': []}
        ref_dict_for_eval[top1_item['question_id']]  = {'question':top1_item['question'],'question_type': top1_item['question_type'],\
            'answers': top1_item['answers'],'entity_answers': [[]],'yesno_answers': []}
            
    mrc.bert.metric.mrc_eval.evaluate(pred_dict_for_eval,ref_dict_for_eval)


class Factory():
    NAME2CLS = {}
    def __init__(self):
        pass
    @classmethod
    def from_dict(cls,d):
        _cls = cls.NAME2CLS[d['class']]
        return _cls(**d['kwargs'])
    @classmethod
    def from_config_path(cls,path,**kwargs):
        config = load_json_config(path,**kwargs)
        return cls.from_config(config,**kwargs)
    @classmethod
    def from_config(cls,config,**kwargs):
        _cls = cls.NAME2CLS[config.CLASS]
        return _cls(config,**kwargs)
    @classmethod
    def from_exp_name(cls,exp_name,**kwargs):
        from .experiment import Experiment
        config =  Experiment(exp_name).config
        return cls.from_config(config,**kwargs)


class RecordGrouper():
    def __init__(self,records):
        self.records = records

    @classmethod
    def from_group_dict(cls,group_key,structured):
        records = []
        for k,l in structured.items():
            for v in l:
                v.update({group_key:k})
                records.append(v)
        return cls(records)
    
    def group(self,field_name):
        df = pd.DataFrame.from_records(self.records)
        gp = df.groupby(field_name).apply(lambda x:x.to_dict('records')).to_dict()
        return gp

    def group_sort(self,group_key,sort_key,k=None):
        df = pd.DataFrame.from_records(self.records)
        gp = df.groupby(group_key).apply(lambda x:x.sort_values(by=sort_key, ascending=False).head(k if k is not None else len(x) ).to_dict('records')).to_dict()
        return gp
            
    def to_records(self):
        l = pd.DataFrame.from_records(self.records).to_dict('records')
        return l


def jsonl_reader(path):
    with open(path,'r',encoding='utf-8') as f:
        for line in f :
            json_obj = json.loads(line.strip(),encoding='utf-8')
            yield json_obj




# convert rawfields to dict
def torchtext_batch_to_dictlist(batch):
    d = {}
    fields_names = list(batch.fields)
    d = { k:getattr(batch,k) for k in fields_names if not isinstance(getattr(batch,k),torch.Tensor)}
    l = pd.DataFrame(d).to_dict('records')
    return l
    



def group_dict_list(dictlist,key,apply_fn=None):
    ret = {}
    for obj in dictlist:
        value = obj[key]
        if value not in ret:
            ret[value] = []
        ret[value].append(obj)
    if apply_fn is None:
        return ret
    for key,v in ret.items():
        ret[key] = apply_fn(v)
    return ret


def group_tuples(tuple_list,item_index,contains_key=False):
    table = {}
    for t in tuple_list:
        key = t[item_index]
        if key not in table:
            table[key] = []
        if contains_key:
            item = t
        else:
            item = tuple(( x for i,x in enumerate(t) if i!=item_index))
        table[key].append(item)
    return table

def tuple2dict(tuples,key_names):
    convert_fn = lambda t: { key:item   for item,key in zip(t,key_names)} 
    if isinstance(tuples,tuple):
        assert len(tuples) == len(key_names)
        return convert_fn(tuples)
    else:
        assert len(tuples[0]) == len(key_names)
    return [  convert_fn(tup)  for tup in tuples]

def load_json_config(path,to_attr=True,**extra_attr):
    with open(path,'r',encoding='utf-8') as f:
        data = json.load(f)
    data.update(extra_attr)
    ret  = data
    if to_attr:
        ret = SimpleNamespace(**data)
    return ret

def get_default_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def tests():
    res = tuple2dict((1,2,3),["a","b","c"])
    print(res)

