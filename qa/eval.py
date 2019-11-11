import torch
from rank.util import group_tuples
from rank.datautil import load_examples_from_scratch
from qa.ranker import RankerFactory
from qa.reader import ReaderFactory
from qa.judger import MaxAllJudger,LambdaJudger,TopKJudger
from qa.para_select import ParagraphSelectorFactory
from common.util import  group_dict_list,RecordGrouper,evaluate_mrc_bidaf
from dataloader.dureader import DureaderLoader,BertRCDataset,BertRankDataset
from dataloader import chiuteacher,demo_files
from io import StringIO
import os


class QARankedListFormater():
    def __init__(self,ranked_dict_list):
        self.results = ranked_dict_list
    def format_result(self,gold_info=False,extra_fields=[]):
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
                for field in extra_fields:
                    if field in pred:
                        buf.write('\t\t%s is %s\n'%(field,pred[field]))
                buf.write('\t\tpassage is %s\n\n'%(pred['passage'][0:500]))
                buf.write('\t\textract (score : %.3f):\n%s\n'%(pred['span_score'],pred['span']))
                buf.write('\t\t##'*10+'\n')
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

def accuracy(rank_dict):
    tot = 0
    cnt = 0
    for q,v in rank_dict.items():
        if len(v) == 0:
             continue
        acc =0 
        for pred in v:
            if pred['rank_score']>=0.5 and pred['label']==1:
                acc+=1
            elif  pred['rank_score']<0.5 and pred['label']==0:
                acc+=1
        tot+= acc /len(v)
        cnt+=1
    return tot/cnt

def evaluate_chiu_rank(ranker_exp_name):
    examples = chiuteacher.read_qa_txt('./data/chiuteacher/問題和答案_分類1.csv')
    print('%d chiu examples'%(len(examples)))
    pairwise_examples  =  chiuteacher.generate_pairwise_examples(examples,sample_k=9)
    records = [dict(zip(['question','passage','label'], values)) for values in pairwise_examples ]
    print('%d chiu pairwise records'%(len(records)))

    ranker =   RankerFactory.from_exp_name(ranker_exp_name)
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






def evaluate_dureader_ranker(paths,ranker,batch_size=64,print_detail=True):
    if type(ranker)==str:
        ranker = RankerFactory.from_exp_name(ranker)
    loader = DureaderLoader(paths,'most_related_para',sample_fields=['question','question_id','answer_docs'])
    samples_to_evaluate = []
    for sample in loader.sample_list:
        if len(sample['answer_docs']) == 0:
            continue
        label = 0
        if sample['doc_id'] == sample['answer_docs'][0]:
            label = 1
        sample['label'] = label
        samples_to_evaluate.append(sample)
    rank_results = ranker.evaluate_on_records(samples_to_evaluate,batch_size=batch_size)
    print( len(rank_results))
    sorted_results = RecordGrouper(rank_results).group_sort('question','rank_score',50)
    if print_detail:
        for k,v in list(sorted_results.items())[0:2]:
            print('question:')
            print(k)
            print('document number [%d] answer doc [%d]'%(len(v),v[0]['answer_docs'][0]))
            for x in v[0:10]:
                print('\t\t label is [%d] doc_id is [%d]'%(x['label'],x['doc_id']))
                print('\t\t'+x['passage'][0:100])
                print('\t\t %.3f'%(x['rank_score']))
                print('# #'*10)
                print('\n')
    print('precision is ')
    print(precision(sorted_results,k=1))
    print('recall is ')
    print(recall(sorted_results,k=1))
    print('accuracy is')
    print(accuracy(sorted_results))


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

def show_mrc_prediction_for_collected_qa_dataset(paths,outpath,reader_name,para_selection_method,decoder_dict=None):
    print('show_prediction collected_qa_dataset')
    selector = ParagraphSelectorFactory.create_selector(para_selection_method)
    loader = DureaderLoader(paths,None,sample_fields=['question','answers','qid'],doc_fields=['title'])
    sample_list = []
    rank_results = {} 
    for qid,v in RecordGrouper(loader.sample_list).group('qid').items():
        ranked_list = selector.evaluate_scores(v)
        rank_results[qid] = RecordGrouper(ranked_list).group_sort('qid','rank_score',10)[qid]
        #aaa = selector.select_top_k_each_doc(ranked_list)
        print('length:%d'%(len(rank_results[qid])))
        sample_list.extend(rank_results[qid])
    rank_file_path = '%s/%s.rank.txt'%(os.path.dirname(outpath),os.path.splitext(os.path.basename(outpath))[0])
    with open(rank_file_path,'w',encoding='utf-8') as f:
        for k,v in rank_results.items():
            print('question:',file=f)
            print(v[0]['question'],file=f)
            for x in v[0:10]:
                print('\t\t answer_doc[%d] : %s'%(x["doc_id"],x["title"]),file=f)
                print('\t\t'+x['passage'],file=f)
                print('\t\t %.3f'%(x['rank_score']),file=f)
                print('# #'*10,file=f)
                print('\n',file=f)   
    print('reading...')
    reader = ReaderFactory.from_exp_name(reader_name,decoder_dict=decoder_dict)
    _preds = reader.evaluate_on_records(sample_list,batch_size=128)
    _preds = group_dict_list(_preds,'qid')
    judger = LambdaJudger(2,lambda x: x['rank_score']+x['span_score'])
    #judger = TopKJudger(k=2)
    #pred_answers  = TopKJudger(k=2).judge(_preds)
    pred_answers  = judger.judge(_preds)
    for pred in _preds:
        pred['title_score'] = 0
        
    pred_answer_list = RecordGrouper.from_group_dict('qid',pred_answers).records
    print('result len %d'%(len(pred_answer_list)))
    print('bidaf evaluation')
    ranked_list_formatter = QARankedListFormater(pred_answer_list)
    formated_result = ranked_list_formatter.format_result(extra_fields=['label','title'])
    with open(outpath,'w',encoding='utf-8') as f:
        #f.write('experiment settings\n')
        #f.write('reader_exp_name : %s\n'%(reader_exp_name))
        #f.write('para_selection_method : %s\n'%(str(para_selection_method)))
        #f.write('decoder : %s\n'%(str(decoder_dict)))
        #f.write('##'*20)
        #f.write('Content:\n\n')
        f.write(formated_result)



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





if __name__ == '__main__':
    DUREADER_DEV_ALL = ['./data/devset/search.dev.json','./data/devset/zhidao.dev.json']
    DEBUG_FILE = ['./data/demo/devset/search.dev.2.json']
    #evaluate_chiu_rank('pointwise/answer_doc')


    #test_dureader_bert_rc(DEBUG_FILE,'reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})

    #test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})
    #test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',{'class':'bert_ranker','kwargs':{'ranker_name':'pointwise/answer_doc'}},decoder_dict={'class':'default','kwargs':{'k':1}})
    #test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',para_selection_method={'class':'tfidf','kwargs':{}},decoder_dict={'class':'default','kwargs':{'k':1}})


    #show_prediction_for_dureader('./data/demo/devset/search.dev.json','prediction.txt','reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})
    #show_prediction_for_dureader('./data/demo/devset/search.dev.json','prediction.txt','reader/bert_default',para_selection_method={'class':'bert_ranker','kwargs':{'ranker':'pointwise/answer_doc'}},decoder_dict={'class':'default','kwargs':{'k':1}})
    #show_prediction_for_demo_examples('reader/bert_default',decoder_dict={'class':'default','kwargs':{'k':1}},out_path='bert_default_k=1.txt')
    #show_prediction_for_demo_examples('reader/bert_default',decoder_dict={'class':'default','kwargs':{'k':2}},out_path='bert_default_k=2.txt')


    #evaluate_dureader_ranker('./data/demo/devset/search.dev.2.json','pointwise/answer_doc')
    #show_mrc_prediction_for_collected_qa_dataset('./data/debug.paragraphs.jsonl','debug_mrc_prediction.txt','reader/bert_default',para_selection_method={'class':'bert_ranker','kwargs':{'ranker':'pointwise/answer_doc'}},decoder_dict={'class':'default','kwargs':{'k':1}})
    #show_mrc_prediction_for_collected_qa_dataset('./data/chiu_question.paragraphs.jsonl','chiu_mrc_prediction.txt','reader/bert_default',para_selection_method={'class':'bert_ranker','kwargs':{'ranker':'pointwise/answer_doc'}},decoder_dict={'class':'default','kwargs':{'k':1}})
    show_mrc_prediction_for_collected_qa_dataset('./data/chiu_common_health.paragraphs.ptag.jsonl','./chiu_common_health_mrc_prediction.txt','reader/bert_default',para_selection_method={'class':'bert_ranker','kwargs':{'ranker':'pointwise/answer_doc','k':2}},decoder_dict={'class':'default','kwargs':{'k':2}})
    #show_mrc_prediction_for_collected_qa_dataset('./tmp.txt','./chiu_common_health_mrc_prediction.txt','reader/bert_default',para_selection_method={'class':'bert_ranker','kwargs':{'ranker':'pointwise/answer_doc','k':2}},decoder_dict={'class':'default','kwargs':{'k':1}})
