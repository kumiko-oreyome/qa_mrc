import functools
import torch
import itertools
from rank.util import predict_on_batch,aggregate_prediction,sort_preidction_by_score,model_factory
from rank.datautil  import load_examples_from_scratch,generate_bert_pointwise_input,BatchIter,numeralize_fucntion_factory
from common.util import tuple2dict,load_json_config,torchtext_batch_to_dictlist,Factory
from common.experiment import Experiment
import pandas as pd
from collections import Counter
from common.textutil import Tokenizer,Tfidf
from dataloader import dureader




class TfIdfRanker():
    def __init__(self,corpus_path_list=['./data/trainset/zhidao.train.json']):
        self.corpus_path_list = corpus_path_list
        samples = dureader.DureaderLoader(self.corpus_path_list,'most_related_para',sample_fields=[]).sample_list
        passages = list(map(lambda x:x['passage'],samples))
        self.tokenizer =  Tokenizer()
        self.tfidf = Tfidf(passages,self.tokenizer.tokenize,corpus_path='./data/tfidf_ranker_corpus')

    def rank(self,example_dict):
        pass

    def evaluate_on_records(self,record_list):
        for record in record_list:
            question = record['question']
            passage =  record['passage']
            score = self.tfidf.cosine_similarity(question,passage)
            record['rank_score'] = score
        return record_list

    def match_score(self,question_tokens,passage_tokens):
        common_with_question = Counter(passage_tokens) & Counter(question_tokens)
        correct_preds = sum(common_with_question.values())
        if correct_preds == 0:
            recall_wrt_question = 0
        else:
            recall_wrt_question = float(correct_preds) / len(question_tokens)
        return recall_wrt_question


class WordMatchRanker():
    def __init__(self,k=1):
        self.k = k
        self.tokenizer =  Tokenizer()
    def rank(self,example_dict):
        pass
    def evaluate_on_records(self,record_list):
        for record in record_list:
            question = record['question']
            passage =  record['passage']
            question_tokens = self.tokenizer.tokenize(question)
            passage_tokens = self.tokenizer.tokenize(passage)
            score = self.match_score(question_tokens,passage_tokens)
            record['rank_score'] = score
        return record_list

    def match_score(self,question_tokens,passage_tokens):
        common_with_question = Counter(passage_tokens) & Counter(question_tokens)
        correct_preds = sum(common_with_question.values())
        if correct_preds == 0:
            recall_wrt_question = 0
        else:
            recall_wrt_question = float(correct_preds) / len(question_tokens)
        return recall_wrt_question




class BertPointwiseRanker():
    def __init__(self,config,eval_flag=True,device=None):
        self.config = config
        self.model,self.tokenizer,self.device  = model_factory(config.BERT_SERIALIZATION_DIR,device=device)
        self.model.load_state_dict(torch.load(config.MODEL_PATH,map_location=self.device))   
        if eval_flag:
            self.model.eval()
        _num_fn = numeralize_fucntion_factory(config.NUM_FN_NAME)
        self.numeralize_fn = functools.partial(_num_fn,max_seq_len=config.MAX_SEQ_LEN ,max_passage_len=config.MAX_PASSAGE_LEN ,\
            tokenizer=self.tokenizer,device=self.device)


    def rank(self,example_dict,batch_size=16):
        #examples = [ [(q,dct['passage'])  for dct in passage_list]  for q,passage_list in example_dict.items() ]
        #examples = list(itertools.chain(*examples))
        examples = []
        for q,passage_list in example_dict.items():
            examples.extend( [(q,d['passage']) for d in passage_list] )
        batch_iter = BatchIter(examples,batch_size,self.numeralize_fn)
        predictions = predict_on_batch(self.model,batch_iter,sigmoid=True)
        result = aggregate_prediction(examples,predictions,labels=None,sort=True)
        sorted_dict = sort_preidction_by_score(result,attach_score=True)
        ret = {}
        for q, tuple_list in sorted_dict.items():
            pred_dict = tuple2dict(tuple_list,['question','passage','rank_score'])
            ret[q] = pred_dict
        return ret


    def evaluate_on_records(self,record_list,batch_size=128):
        iterator = self.get_batchiter(record_list,batch_size)
        return self.evaluate_on_batch(iterator)

    def get_batchiter(self,record_list,batch_size=64):
        rank_dataset = dureader.BertRankDataset(record_list,self.config.BERT_SERIALIZATION_DIR,self.config.MAX_PASSAGE_LEN,self.config.MAX_SEQ_LEN)
        iterator =  rank_dataset.make_batchiter(batch_size=batch_size)
        return iterator


    def evaluate_on_batch(self,iterator):
        with torch.no_grad():
            preds = []
            for  i,batch in enumerate(iterator):
                #if (i+1) % 20 == 0:
                #    print('ranker : evaluate on %d batch'%(i))
                match_scores = self.predict_score_one_batch(batch)
                if match_scores.is_cuda:
                    match_scores = match_scores.cpu()
                match_scores =  match_scores.numpy().tolist()
                batch_dct_list =  torchtext_batch_to_dictlist(batch)
                for j,item_dict in enumerate(batch_dct_list):
                    item_dict.update({'rank_score':match_scores[j]})
                    preds.append(item_dict)
        return  preds

    def predict_score_one_batch(self,batch):
        match_scores = self.model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask)
        match_scores  = torch.nn.Sigmoid()(match_scores) # N,2
        match_scores  =  match_scores[:,1] #  N  ,get positve socre
        match_scores = match_scores
        return match_scores




class RankerFactory(Factory):
    NAME2CLS = {'bert_pointwise':BertPointwiseRanker}
    def __init__(self):
        pass



