
import torch
from rank.datautil import load_examples_from_scratch
from qa.ranker import RankerFactory
from qa.reader import ReaderFactory
from qa.judger import MaxAllJudger,MultiplyJudger
from common.dureader_eval  import  compute_bleu_rouge,normalize
from common.util import  group_dict_list,RecordGrouper,evaluate_mrc_bidaf
from dataloader.dureader import DureaderLoader,BertRCDataset,BertRankDataset
from dataloader import chiuteacher  
import mrc.bert.metric.mrc_eval
import pandas as pd
import numpy as np
import itertools 
import jieba as jb
from common.textutil import Tokenizer
from torch.optim import SGD

##ls
# load data
# fileds answer_doc question question_id passages answers
# ranker prediction --> get rank scores on all records
# normalize scores on same question  to probality (policy) and select a passage by policy


class MetricTracer():
    def __init__(self):
        self.tot = 0
        self.n = 0
    def zero(self):
        self.tot = 0
        self.n = 0
    def add_record(self,record):
        self.tot+=record
        self.n+=1
    def avg(self):
        return self.tot/self.n
    def print(self,prefix=''):
        print('%s%.3f'%(prefix,self.avg()))

def transform_policy_score(ranker_results,field_name='rank_score'):
    grouper = RecordGrouper(ranker_results)
    ret = []
    for _,v in grouper.group('question_id').items():
        df = pd.DataFrame.from_records(v)
        df['policy_score']= df[field_name]/df[field_name].sum()
        ret.extend(df.to_dict('records'))
    return ret
       

def text_overlap_precision(text_words,answer_words):
    s1 = set(text_words)
    if len(s1) == 0:
        return 0
    s2 = set(answer_words)
    m = s1 & s2
    return len(m)/len(s1)
    
def text_overlap_recall(text_words,answer_words):
    s1 = set(text_words)
    s2 = set(answer_words)
    m = s1 & s2
    return len(m)/len(s2)



# rc reward
def reward_function_word_overlap(prediction,ground_truth):
    precision = text_overlap_precision(prediction,ground_truth)
    if precision == 0:
        return -1
    recall = text_overlap_recall(prediction,ground_truth)
    f1 = 2*precision*recall/(precision+recall)
    if f1 >2:
        f1 = 2
    return f1


def policy_gradient(rewards,probs):
    loss =  rewards*torch.log(probs)
    return torch.mean(loss)

class PolicySampleRanker():
    def __init__(self,records,score_field='policy_score'):
        self.records = records
        self.score_field = score_field
    def sample_per_question(self,k=1,sample_field='question_id'):
        df = pd.DataFrame.from_records(self.records)
        aaa = df.groupby(sample_field).apply(lambda x:self._sample_lambda(x,k).to_dict('records')).tolist()
        l =  list(itertools.chain(*aaa))
        return l

    def _sample_lambda(self,group,k):
        probs = group[self.score_field].tolist()
        indexs = np.random.choice(len(probs),size=k,replace=False, p=probs)
        return group.iloc[indexs]


if __name__ == '__main__':
    TRAIN_PATH = "./data/demo/devset/search.dev.2.json"
    DEV_PATH = "./data/demo/devset/search.dev.2.json"
    READER_EXP_NAME = 'reader/bert_default'
    RANKER_EXP_NAME = 'pointwise/answer_doc'
    EPOCH = 100
    train_loader = DureaderLoader(TRAIN_PATH ,'most_related_para',sample_fields=['question','answers','question_id','question_type','answer_docs','answer_spans'])
    dev_loader = DureaderLoader(DEV_PATH ,'most_related_para',sample_fields=['question','answers','question_id','question_type','answer_docs'])
    ranker = RankerFactory.from_exp_name(RANKER_EXP_NAME,eval_flag=False)
    reader = ReaderFactory.from_exp_name(READER_EXP_NAME,eval_flag=False)
    tokenizer =  Tokenizer()
    reader_optimizer =  SGD(reader.model.parameters(), lr=0.00001, momentum=0.9)
    ranker_optimizer = SGD(ranker.model.parameters(), lr=0.00001, momentum=0.9)
    BATCH_SIZE = 64

    ranker_results = ranker.evaluate_on_records(train_loader.sample_list)
    results_with_poicy_scores = transform_policy_score(ranker_results)
    policy = PolicySampleRanker(results_with_poicy_scores)
    sampled_records = policy.sample_per_question()
    for epcoch in range(EPOCH):
        train_batch = reader.get_batchiter(sampled_records,train_flag=True,batch_size=BATCH_SIZE)
        reader_loss,ranker_loss,reward_tracer = MetricTracer(),MetricTracer(),MetricTracer()
        for  i,batch in enumerate(train_batch):
            if i % 20 == 0 and i!=0:
                print('reinfroce loop evaluate on %d batch'%(i))
                reader_loss.print()
            start_pos,end_pos = tuple(zip(*batch.answer_span))
            start_pos,end_pos = torch.tensor(start_pos,device=reader.device, dtype=torch.long),torch.tensor(end_pos,device=reader.device, dtype=torch.long)
            loss,start_logits, end_logits  = reader.model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask, start_positions=start_pos, end_positions=end_pos)
            with torch.no_grad():
                reader_predictions = reader.decode_batch(start_logits.detach(), end_logits.detach(),batch)
                reader_predictions  = [{'question':d['question'],'passage':d['passage'],'answers':d['answers'],'rank_score':d['rank_score'],'span':d['span']} for d in reader_predictions]
            loss.backward()
            reader_optimizer.step()
            reader_optimizer.zero_grad()
            reader_loss.add_record(loss.item())
            ##calculate rewards
            
            for pred in reader_predictions:
                pred_tokens = tokenizer.tokenize(pred['span'])
                reward = max([ reward_function_word_overlap(pred_tokens,tokenizer.tokenize(answer)) for answer in pred['answers']])
                pred['reward'] = reward
                reward_tracer.add_record(reward)


            ##prediction on ranker (with grad)
            for  ranker_batch in ranker.get_batchiter(reader_predictions,batch_size=BATCH_SIZE):
                ##prediction on ranker (with grad)
                rewards = torch.tensor(ranker_batch.reward,device=ranker.device,dtype=torch.float)
                ranker_probs = ranker.predict_score_one_batch(batch)
                loss = policy_gradient(rewards,ranker_probs)
                loss.backward()
                ranker_optimizer.step()
                ranker_optimizer.zero_grad()
                ranker_loss.add_record(loss.item())
        print('EPOCH: %d'%(epcoch))
        reader_loss.print('reader avg loss:')
        ranker_loss.print('ranker avg loss:')
        reward_tracer.print('reward avg :')
        print('metrics')
        #here must let ranker to select paragraph to evaluate the effect of policy gradient
        _preds = reader.evaluate_on_batch(reader.get_batchiter(dev_loader.sample_list,batch_size=BATCH_SIZE))
        _preds = group_dict_list(_preds,'question_id')
        pred_answers  = MaxAllJudger().judge(_preds)
        evaluate_mrc_bidaf(pred_answers)
        print('- - '*20)





