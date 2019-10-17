import torch
from rank.util import group_tuples
from rank.datautil import load_examples_from_scratch
from qa.ranker import RankerFactory
from qa.reader import ReaderFactory
from qa.judger import MaxAllJudger,MultiplyJudger
from common.util import  group_dict_list,RecordGrouper,evaluate_mrc_bidaf
from dataloader.dureader import DureaderLoader,BertRCDataset,BertRankDataset
from dataloader import chiuteacher,demo_files
from io import StringIO


class QARankedListFormater():
    def __init__(self,ranked_dict_list):
        self.results = ranked_dict_list
    def format_result(self,gold_info=False):
        grouper_dict = RecordGrouper(self.results).group('question')
        buf = StringIO()
        for q,v in grouper_dict.items():
            buf.write('question:\n')
            buf.write(q+'\n')
            #if gold_info:
            #    buf.write('dureader ground truth information:\n')
            #    buf.write('answer passage is \n')   
            #    buf.write('answer is \n')
            buf.write('prediction answers:\n')
            for pred in v:
                buf.write('\t\tpassage is %s\n\n'%(pred['passage'][0:500]))
                buf.write('\t\textract (score : %.3f):\n%s\n'%(pred['span_score'],pred['span']))
                buf.write('\t\t##'*10)
        return buf.getvalue()
                
        




# evaluate by the method of dureader bert probject
def evaluate_mrc_bert(pred_answers):
    import mrc.bert.metric.mrc_eval
    pred_dict_for_eval = {}
    ref_dict_for_eval  = {}
    for _,v in pred_answers.items():
        top1_item = v[0]
        pred_dict_for_eval[top1_item['question_id']] = {'question':top1_item['question'],'question_type': top1_item['question_type'],\
            'answers': [top1_item['span']],'entity_answers': [[]],'yesno_answers': []}
        ref_dict_for_eval[top1_item['question_id']]  = {'question':top1_item['question'],'question_type': top1_item['question_type'],\
            'answers': top1_item['answers'],'entity_answers': [[]],'yesno_answers': []}
            
    mrc.bert.metric.mrc_eval.evaluate(pred_dict_for_eval,ref_dict_for_eval)


def precision(rank_dict,k=1):
    tot = 0
    cnt = 0
    for q,v in rank_dict.items():
        l = rank_dict[q][0:k]
        acc = sum([ record['label'] for record in l ])
        tot+= acc/len(l)
        cnt+=1
    return tot/cnt


def recall(rank_dict,k=1):
    tot = 0
    cnt = 0
    for q,v in rank_dict.items():
        l = rank_dict[q][0:k]
        acc = sum([ record['label'] for record in l ])
        tot+= acc/sum([ record['label'] for record in v ])
        cnt+=1
    return tot/cnt

def evaluate_chiu_rank(ranker_exp_name):
    examples = chiuteacher.read_qa_txt('./data/chiuteacher/問題和答案_分類1.csv')
    print('%d chiu examples'%(len(examples)))
    pairwise_examples  =  chiuteacher.generate_pairwise_examples(examples,sample_k=9)
    records = [dict(zip(['question','passage','label'], values)) for values in pairwise_examples ]
    print('%d chiu pairwise records'%(len(records)))

    ranker =   RankerFactory.from_exp_name(ranker_exp_name,RANKER_CLASS='bert_pointwise')
    ranker_config = ranker.config
    rank_dataset = BertRankDataset(records,ranker_config.BERT_SERIALIZATION_DIR,ranker_config.MAX_PASSAGE_LEN,ranker_config.MAX_SEQ_LEN)
    iterator =  rank_dataset.make_batchiter(batch_size=128)
    rank_results = ranker.evaluate_on_batch(iterator)
    
    print( len(rank_results))
    sorted_results = RecordGrouper(rank_results).group_sort('question','rank_score',50)
    for k,v in sorted_results.items():
        print('question:')
        print(k)
        for x in v[0:10]:
            print('\t\t'+x['passage'][0:100])
            print('\t\t %.3f'%(x['rank_score']))
            print('# #'*10)
            print('\n')
    print('precision is ')
    print(precision(sorted_results,k=1))
    print('recall is ')
    print(recall(sorted_results,k=1))







def test_dureader_bert_rc_with_ranker_XXX(test_path,ranker_exp_name,reader_exp_name,para_selection_method,judge_method,decode_policy='greedy',batch_size=128):
    print('test_dureader_bert_rc loading samples...')
    loader = DureaderLoader(test_path,para_selection_method,sample_fields=['question','answers','question_id','question_type'])

    ranker =   RankerFactory.from_exp_name(ranker_exp_name,RANKER_CLASS='bert_pointwise')
    ranker_config = ranker.config

    rank_dataset = BertRankDataset(loader.sample_list,ranker_config.BERT_SERIALIZATION_DIR,ranker_config.MAX_PASSAGE_LEN,ranker_config.MAX_SEQ_LEN)
    iterator =  rank_dataset.make_batchiter(batch_size=batch_size)
    rank_results = ranker.evaluate_on_batch(iterator)

    sorted_results = RecordGrouper(rank_results).group_sort('question','rank_score')
    #for q,v  in sorted_results.items():
    #    print(q)
    #    print('- - -')
    #    for obj in v:
    #        print(obj['passage'][0:100])
    #        print(obj['rank_score'])
    #        print('**'*10)
    #ranked_sample_list = RecordGrouper.from_group_dict('question',sorted_results).to_records()
    #print(rank_results[0:2])
    reader = ReaderFactory.from_exp_name(reader_exp_name,READER_CLASS='bert_reader',decode_policy='greedy')
    reader_config = reader.config

    dataset  = BertRCDataset(rank_results,reader_config.MAX_QUERY_LEN,reader_config.MAX_SEQ_LEN,device=reader.device)
    print('make batch')
    iterator = dataset.make_batchiter(batch_size=batch_size)
    extracted_answer = reader.evaluate_on_batch(iterator)
    extracted_answer_dict = RecordGrouper(extracted_answer).group('question_id')
    if judge_method is 'max_all':
        pred_answers  = MaxAllJudger().judge(extracted_answer_dict)
    elif judge_method is 'multiply':
        pred_answers  = MultiplyJudger().judge(extracted_answer_dict)
    else:
        assert False
    
    #for q,v  in pred_answers.items():
    #    print(q)
    #    print('- - -')
    #    for obj in v:
    #        print(obj['span'])
    #        print(obj['rank_score'])
    #        print(obj['span_score'])
    #        print('**'*10)
    evaluate_mrc_bert(pred_answers)



def test_dureader_bert_rc(test_path,reader_exp_name,para_selection_method,decoder_dict=None):
    print('test_dureader_bert_rc loading samples...')
    loader = DureaderLoader(test_path,para_selection_method,sample_fields=['question','answers','question_id','question_type'])
    sample_list = loader.sample_list
    reader = ReaderFactory.from_exp_name(reader_exp_name,decoder_dict=decoder_dict)
    _preds = reader.evaluate_on_records(sample_list,batch_size=128)
    _preds = group_dict_list(_preds,'question_id')
    pred_answers  = MaxAllJudger().judge(_preds)
    print('bidaf evaluation')
    evaluate_mrc_bidaf(pred_answers)


def show_prediction_for_dureader(paths,outpath,reader_exp_name,para_selection_method,decoder_dict=None):
    print('show_prediction_for_dureader')
    loader = DureaderLoader(paths,para_selection_method,sample_fields=['question','answers','question_id','question_type'])
    sample_list = loader.sample_list
    reader = ReaderFactory.from_exp_name(reader_exp_name,decoder_dict=decoder_dict)
    _preds = reader.evaluate_on_records(sample_list,batch_size=128)
    _preds = group_dict_list(_preds,'question_id')
    pred_answers  = MaxAllJudger().judge(_preds)
    pred_answer_list = RecordGrouper.from_group_dict('question_id',pred_answers).records
    print('bidaf evaluation')
    ranked_list_formatter = QARankedListFormater(pred_answer_list)
    formated_result = ranked_list_formatter.format_result()
    with open(outpath,'w',encoding='utf-8') as f:
        f.write('experiment settings\n')
        f.write('reader_exp_name : %s\n'%(reader_exp_name))
        f.write('para_selection_method : %s\n'%(str(para_selection_method)))
        f.write('decoder : %s\n'%(str(decoder_dict)))
        f.write('##'*20)
        f.write('Content:\n\n')
        f.write(formated_result)

def show_mrc_prediction_for_collected_qa_dataset(test_path,reader_name):
    pass


def show_prediction_for_demo_examples(reader_name,decoder_dict,test_path='./data/examples.txt',out_path='demo_mrc.txt'):
    samples = demo_files.read_from_demo_txt_file(test_path)
    reader = ReaderFactory.from_exp_name(reader_name,decoder_dict=decoder_dict)
    _preds = reader.evaluate_on_records(samples,batch_size=128)
    f = open(out_path,'w',encoding='utf-8')
    for sample in _preds:
        print('Question',file=f)
        print(sample['question'],file=f)
        print('Passage',file=f)
        print('%s'%(sample['passage']),file=f)
        print('--'*20,file=f)
        print('Answer:',file=f)
        print('%s'%(sample['span']),file=f)
        print('# # #'*20,file=f)





DUREADER_DEV_ALL = ['./data/devset/search.dev.json','./data/devset/zhidao.dev.json']
DEBUG_FILE = ['./data/demo/devset/search.dev.2.json']
#evaluate_chiu_rank('pointwise/answer_doc')


#test_dureader_bert_rc(DEBUG_FILE,'reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})

#test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})
#test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',{'class':'bert_ranker','kwargs':{'ranker_name':'pointwise/answer_doc'}},decoder_dict={'class':'default','kwargs':{'k':1}})
#test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',para_selection_method={'class':'tfidf','kwargs':{}},decoder_dict={'class':'default','kwargs':{'k':1}})


#show_prediction_for_dureader('./data/demo/devset/search.dev.json','prediction.txt','reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})
#show_prediction_for_dureader('./data/demo/devset/search.dev.json','prediction.txt','reader/bert_default',para_selection_method={'class':'bert_ranker','kwargs':{'ranker_name':'pointwise/answer_doc'}},decoder_dict={'class':'default','kwargs':{'k':1}})
show_prediction_for_demo_examples('reader/bert_default',decoder_dict={'class':'default','kwargs':{'k':1}})


