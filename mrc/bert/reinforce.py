import torch
from .util import preprocessing_charspan
from rank.datautil import load_examples_from_scratch
from qa.ranker import RankerFactory
from qa.reader import ReaderFactory
from qa.judger import MaxAllJudger
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
from torch.optim.lr_scheduler import StepLR

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





#def adjust_learning_rate(optimizer, epoch):
#    lr = args.lr * (0.1 ** (epoch // 30))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = l

def transform_policy_score(ranker_results,field_name='rank_score'):
    grouper = RecordGrouper(ranker_results)
    ret = []
    for _,v in grouper.group('question_id').items():
        df = pd.DataFrame.from_records(v)
        assert df[field_name].sum() > 0
        assert np.all(df[field_name]>0)
        df['policy_score']= df[field_name]/df[field_name].sum()
        records = df.to_dict('records')
        answer_score = [record for record in records if record['answer_docs'][0]==record["doc_id"]][0]['policy_score']
        for r in records:
            r['answer_score'] =  answer_score
        ret.extend(records)
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
    assert torch.all(probs>0)
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
    def sample_per_question(self,k=1,sample_field='question_id',keep_ground_truth=True):
        df = pd.DataFrame.from_records(self.records)
        aaa = df.groupby(sample_field).apply(lambda x:self._sample_lambda(x,k)).tolist()
        l =  list(itertools.chain(*aaa))
        return l

    def _sample_lambda(self,group,k,keep_ground_truth=True):
        if not keep_ground_truth:
            candidates = group.sort_values(by=self.score_field,ascendind=False)[0:5]
            probs = candidates[self.score_field].tolist()
            indexs = np.random.choice(len(probs),size=k,replace=True, p=probs)
            return candidates.iloc[indexs].to_dict('records')
        else:
            ret = []
            records = group.to_dict('records')
            pos_sample = [sample for sample in records if sample["doc_id"] == sample['answer_docs'][0]]
            assert len(pos_sample)==1
            pos_sample = pos_sample[0]
            neg_samples = [sample for sample in records if sample["doc_id"] != sample['answer_docs'][0]]
            neg_samples = sorted(neg_samples,key=lambda x:x[self.score_field],reverse=True)[0:3]
            neg_samples = np.random.choice(neg_samples,size=min(k,len(neg_samples)),replace=False)
            ret.append(pos_sample)
            ret.extend(neg_samples)
            return ret


if __name__ == '__main__':
    # evaluate ranker
    from qa.eval import evaluate_dureader_ranker
    experiment = Experiment('reader/pg')
    #TRAIN_PATH = ["./data/trainset/search.train.json","./data/trainset/zhidao.train.json"]
    TRAIN_PATH = ["./data/trainset/search.train.json"]
    DEV_PATH = "./data/devset/search.dev.json"
    #TRAIN_PATH = "./data/demo/devset/search.dev.json"
    #DEV_PATH = "./data/demo/devset/search.dev.json"
    READER_EXP_NAME = 'reader/bert_default'
    RANKER_EXP_NAME = 'pointwise/answer_doc'
    EPOCH = 10
    print_detail = False
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
    reader_optimizer =  SGD(reader.model.parameters(), lr=0.0001, momentum=0.9)
    ranker_optimizer = SGD(ranker.model.parameters(), lr=0.0002, momentum=0.9)
    lr_scheduler = StepLR(ranker_optimizer, step_size=10, gamma=0.99)
    BATCH_SIZE = 16
    highest_bleu = -1
    print('ranker performance before traning')
    ranker.model = ranker.model.eval()
    evaluate_dureader_ranker(DEV_PATH,ranker,BATCH_SIZE,print_detail=print_detail)


    for epcoch in range(EPOCH):
        ranker.model = ranker.model.eval()
        print('start of epoch %d'%(epcoch))
        reader_loss,ranker_loss,reward_tracer = MetricTracer(),MetricTracer(),MetricTracer()
        print('start training loop')
        for i,rl_samples in enumerate(ReinforceBatchIter(train_loader.sample_list).get_batchiter(BATCH_SIZE*5)):
            if (i+1) % 100 == 0 :
                print('reinfroce loop evaluate on %d batch'%(i))
                reader_loss.print()
            # rl train
            ranker_results = ranker.evaluate_on_records(rl_samples,batch_size=BATCH_SIZE)
            neg_sample_records = ranker_results
            #neg_sample_records = negative_sampleing(ranker_results,k=1)
            results_with_poicy_scores = transform_policy_score(neg_sample_records)
            policy = PolicySampleRanker(results_with_poicy_scores)
            sampled_records = policy.sample_per_question(2)
            #print('sample recoreds')
            #for v in sampled_records:
            #    print('question %s'%(v['question']))
            #    print('label %d'%(1 if v['doc_id']==v['answer_docs'][0] else 0))
            #    print('passage %s'%(v['passage'][0:100]))
            # answer extraction 
            reader_predictions = reader.evaluate_on_records(sampled_records,batch_size=BATCH_SIZE)
            # calculate rewards          
            for pred in reader_predictions:
                 pred_tokens = tokenizer.tokenize(pred['span'])
                 #reward = max([ reward_function_word_overlap(pred_tokens,tokenizer.tokenize(answer)) for answer in pred['answers']])
                 
                 if pred['answer_docs'][0] == pred['doc_id']:
                     reward =  1+pred['policy_score']  
                 else:
                     reward = -1* pred['policy_score']-1
                 pred['reward'] = reward
                 pred['p_mul'] = pred['policy_score']/pred['rank_score']
                 #pred['reward'] = reward-0.5 i
                 #pred['reward'] = reward
                 #if(pred['answer_score']-pred['policy_score'])<0:
                     #pred['reward']+= 3*(pred['answer_score']-pred['policy_score'])
                 # add reward by postive answer score maybe can learn faster
                 #pred['reward']+= 2*(pred['answer_score']-pred['policy_score'])
                 #pred['reward'] = pred['reward']*2
                 reward_tracer.add_record(reward)
                 #print('reward %.3f'%(pred['reward']))
            # policy_gradient
            for  ranker_batch in ranker.get_batchiter(reader_predictions,batch_size=BATCH_SIZE):
                ##prediction on ranker (with grad)
                rewards = torch.tensor(ranker_batch.reward,device=ranker.device,dtype=torch.float)
                pmul_t = torch.tensor(ranker_batch.p_mul).to(ranker.device)
                ranker_probs_1 = ranker.predict_score_one_batch(ranker_batch)
                try:
                    assert torch.all(ranker_probs_1>0)
                except:
                    import pdb;
                    pdb.set_trace()
                    print(ranker_probs_1)
                ranker_probs = ranker_probs_1*pmul_t
                assert torch.all(pmul_t>0)
                #import numpy as np
                #try:
                #    assert np.array_equal(ranker_batch.policy_score , ranker_probs.detach().cpu().numpy())
                #except :
                #    print(ranker_batch.policy_score)
                #    print(ranker_probs.detach().cpu().numpy())

                loss = -1*policy_gradient(rewards,ranker_probs)
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
                    lr_scheduler.step()
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
        evaluate_dureader_ranker(DEV_PATH,ranker,BATCH_SIZE,print_detail=print_detail)
        reader.model = reader.model.train()
        ranker.model = ranker.model.train()
        if evaluate_result['Bleu-4'] >  highest_bleu:
            print('%.3f %.3f'%(evaluate_result['Bleu-4'],highest_bleu))
            highest_bleu = evaluate_result['Bleu-4']
            model_dir = experiment.model_dir
            print('save models with bleu %.3f to %s'%(highest_bleu,model_dir))
            torch.save(ranker.model.state_dict(),'%s/ranker.bin'%(model_dir))
            torch.save(reader.model.state_dict(),'%s/reader.bin'%(model_dir))
        print('- - '*20)