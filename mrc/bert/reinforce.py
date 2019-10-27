
import torch
from .util import preprocessing_charspan
from rank.datautil import load_examples_from_scratch
from qa.ranker import RankerFactory
from qa.reader import ReaderFactory
from qa.judger import MaxAllJudger,MultiplyJudger
from qa.para_select import BertRankerSelector
from common.dureader_eval  import  compute_bleu_rouge,normalize
from common.experiment import Experiment
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
        if self.tot == 0:
            return 0
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

def negative_sampleing(records,k):
    if k is None:
        return records
    grouper = RecordGrouper(records)
    ret = []
    for _,v in grouper.group('question_id').items():
        pos_sample = [sample for sample in v if sample["doc_id"] == sample['answer_docs'][0]]
        assert len(pos_sample)==1
        pos_sample = pos_sample[0]
        neg_samples = [sample for sample in v if sample["doc_id"] != sample['answer_docs'][0]]
        neg_samples = np.random.choice(neg_samples,size=min(k,len(neg_samples)),replace=False)
        ret.append(pos_sample)
        ret.extend(neg_samples)
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
    loss =  rewards*torch.log(probs+0.00000001)
    return torch.mean(loss)


class ReinforceBatchIter():
    def __init__(self,sample_list):
        self.sample_list = sample_list
    def get_batchiter(self,batch_size,sample_field='question_id'):
        df = pd.DataFrame.from_records(self.sample_list)
        ret = []
        for i,(group_name, df_group) in enumerate(df.groupby(sample_field)):
            l = df_group.to_dict('records')
            ret.extend(l)
            if   (i+1)%batch_size==0:
                yield ret
                ret = []
        if len(ret) > 0:
            yield ret





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
    # evaluate ranker
    from qa.eval import evaluate_dureader_ranker
    experiment = Experiment('reader/pg')
    #TRAIN_PATH = ["./data/trainset/search.train.json","./data/trainset/zhidao.train.json"]
    #TRAIN_PATH = ["./data/trainset/search.train.json"]
    #DEV_PATH = "./data/devset/search.dev.json"
    TRAIN_PATH = "./data/demo/devset/search.dev.2.json"
    DEV_PATH = "./data/demo/devset/search.dev.2.json"
    READER_EXP_NAME = 'reader/bert_default'
    RANKER_EXP_NAME = 'pointwise/answer_doc'
    EPOCH = 10

    TRAIN_READER= False

    train_loader = DureaderLoader(TRAIN_PATH ,'most_related_para',sample_fields=['question','answers','question_id','question_type','answer_docs','answer_spans'],\
        doc_fields=['segmented_paragraphs'])
    print('preprocessing span for  train data')
    train_loader.sample_list = list(filter(lambda x:len(x['answers'])>0 and len(x['answer_docs'])>0,train_loader.sample_list ))
    for sample in  train_loader.sample_list:
        if sample["doc_id"] == sample['answer_docs'][0]:
            preprocessing_charspan(sample)
        else:
            sample['char_spans'] = [0,0]
            del sample['answer_spans']
            del sample['segmented_paragraphs']
    print('load ranker')
    ranker = RankerFactory.from_exp_name(experiment.config.ranker_name,eval_flag=False)
    print('load reader')
    reader = ReaderFactory.from_exp_name(experiment.config.reader_name,eval_flag=False)
    tokenizer =  Tokenizer()
    reader_optimizer =  SGD(reader.model.parameters(), lr=0.00001, momentum=0.9)
    ranker_optimizer = SGD(ranker.model.parameters(), lr=0.00001, momentum=0.9)
    BATCH_SIZE = 12

    print('ranker performance before traning')
    ranker.model = ranker.model.eval()
    evaluate_dureader_ranker(DEV_PATH,ranker,BATCH_SIZE,print_detail=False)
    ranker.model = ranker.model.train()


    for epcoch in range(EPOCH):
        print('start of epoch %d'%(epcoch))
        reader_loss,ranker_loss,reward_tracer = MetricTracer(),MetricTracer(),MetricTracer()
        print('start training loop')
        for i,rl_samples in enumerate(ReinforceBatchIter(train_loader.sample_list).get_batchiter(BATCH_SIZE*5)):
            if (i+1) % 100 == 0 :
                print('reinfroce loop evaluate on %d batch'%(i))
                reader_loss.print()
            # rl train
            ranker_results = ranker.evaluate_on_records(rl_samples,batch_size=BATCH_SIZE)
            neg_sample_records = negative_sampleing(ranker_results,k=1)
            results_with_poicy_scores = transform_policy_score(neg_sample_records)
            policy = PolicySampleRanker(results_with_poicy_scores)
            sampled_records = policy.sample_per_question()
            # answer extraction 
            reader_predictions = reader.evaluate_on_records(sampled_records,batch_size=BATCH_SIZE)
            # calculate rewards
            for pred in reader_predictions:
                 pred_tokens = tokenizer.tokenize(pred['span'])
                 reward = max([ reward_function_word_overlap(pred_tokens,tokenizer.tokenize(answer)) for answer in pred['answers']])
                 pred['reward'] = reward
                 reward_tracer.add_record(reward)
            # policy_gradient
            for  ranker_batch in ranker.get_batchiter(reader_predictions,batch_size=BATCH_SIZE):
                ##prediction on ranker (with grad)
                rewards = torch.tensor(ranker_batch.reward,device=ranker.device,dtype=torch.float)
                ranker_probs = ranker.predict_score_one_batch(ranker_batch)
                loss = policy_gradient(rewards,ranker_probs)
                loss.backward()
                ranker_optimizer.step()
                ranker_optimizer.zero_grad()
                ranker_loss.add_record(loss.item())
            #print('supevisely train reader')
            #supevisely train reader
            if TRAIN_READER:
                train_samples = [ sample for sample in rl_samples if sample["doc_id"] == sample['answer_docs'][0]]
                train_batch = reader.get_batchiter(train_samples,train_flag=True,batch_size=4) ###... reader batch size must be small ...
                for batch in train_batch:
                    start_pos,end_pos = tuple(zip(*batch.bert_span))
                    start_pos_t,end_pos_t = torch.tensor(start_pos,device=reader.device, dtype=torch.long),torch.tensor(end_pos,device=reader.device, dtype=torch.long)
                    loss,start_logits, end_logits  = reader.model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask, start_positions=start_pos_t, end_positions=end_pos_t)
                    loss.backward()
                    reader_optimizer.step()
                    reader_optimizer.zero_grad()
                    reader_loss.add_record(loss.item())
        
        print('EPOCH: %d'%(epcoch))
        reader_loss.print('reader avg loss:')
        ranker_loss.print('ranker avg loss:')
        reward_tracer.print('reward avg :')
        print('metrics')
        #here must let ranker to select paragraph to evaluate the effect of policy gradient
        reader.model = reader.model.eval()
        ranker.model = ranker.model.eval()
        para_selector = BertRankerSelector(ranker)
        loader = DureaderLoader(DEV_PATH,para_selector,sample_fields=['question','answers','question_id','question_type'])
        _preds = reader.evaluate_on_records(loader.sample_list,batch_size=BATCH_SIZE)
        _preds = group_dict_list(_preds,'question_id')
        pred_answers  = MaxAllJudger().judge(_preds)
        evaluate_result = evaluate_mrc_bidaf(pred_answers)



        print('evaluate ranker')
        evaluate_dureader_ranker(DEV_PATH,ranker,BATCH_SIZE,print_detail=False)

        reader.model = reader.model.train()
        ranker.model = ranker.model.train()
        higest_bleu = 0.0
        if evaluate_result['Bleu-4'] >  higest_bleu:
            higest_bleu = evaluate_result['Bleu-4']
            model_dir = experiment.model_dir
            print('save models with bleu %.3f to %s'%(evaluate_result['Bleu-4'],model_dir))
            torch.save(ranker.model.state_dict(),'%s/ranker.bin'%(model_dir))
            torch.save(reader.model.state_dict(),'%s/reader.bin'%(model_dir))
        print('- - '*20)





