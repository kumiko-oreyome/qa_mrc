import torch
from rank.util import group_tuples
from rank.datautil import load_examples_from_scratch
from qa.ranker import RankerFactory
from qa.reader import ReaderFactory
from qa.judger import MaxAllJudger,MultiplyJudger
from common.dureader_eval  import  compute_bleu_rouge,normalize
from common.util import  group_dict_list,RecordGrouper
from dataloader.dureader import DureaderLoader,BertRCDataset,BertRankDataset
from dataloader import chiuteacher  
import mrc.bert.metric.mrc_eval


# dureader dataset test

TESTPATH = './'


# evaluate by the method of dureader bert probject
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
    print(compute_bleu_rouge(pred_for_bidaf_eval,ref_dict))
       




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
    reader_config = reader.config

    dataset  = BertRCDataset(sample_list,reader_config.MAX_QUERY_LEN,reader_config.MAX_SEQ_LEN,device=reader.device)
    print('make batch')
    iterator = dataset.make_batchiter(batch_size=128)
    _preds = reader.evaluate_on_batch(iterator)
    _preds = group_dict_list(_preds,'question_id')
    pred_answers  = MaxAllJudger().judge(_preds)

    #evaluate_mrc_bert(pred_answers)


    #answer_dict = group_dict_list(sample_list,'question_id', lambda x: {'answers':normalize([x[0]['answers']])})
    
    #print(answer_dict)
    #print(pred_answers)
    #print('bert evaluation')
    #evaluate_mrc_bert(pred_answers)
    print('bidaf evaluation')
    evaluate_mrc_bidaf(pred_answers)





def evaluate_dureader(test_path,ranker_exp_name,reader_config_path):
    batch_size = 24
    #examples,labels = load_examples_from_scratch(test_path,attach_label='answer_docs')
    loader = DureaderLoader(test_path)
    ranker_input = {}
    for sample in loader.sample_list:
        question,passage = sample['question'],sample['passage']
        if question not in ranker_input:
            ranker_input[question] = []
        ranker_input[question].append({'passage':passage})
    
    ranker =   RankerFactory.from_exp_name(ranker_exp_name,RANKER_CLASS='bert_pointwise')
    print('ranking')
    rank_result = ranker.rank(ranker_input,batch_size=batch_size)
    reader = ReaderFactory.from_exp_name('reader/bert_default',READER_CLASS='bert_reader')
    print('reading')
    reader_result = reader.extract_answer(rank_result,batch_size=batch_size)
    #for q,v in reader_result.items():
    #    print(q)
    #    for x in v:
    #        print(x['span'])
    #        print(x['span_score'])
    #        print(x['rank_score'])
    #    break
    print('evaluating')
    pred_answers  = MaxAllJudger().judge(reader_result)
    answer_dict = loader.aggregate_by_filed('question')
    print(pred_answers)
    #evaluate_dureader_result(pred_answers,answer_dict)
    #for pred, ref in zip(pred_answers, ref_answers):
    #    question_id = ref['question_id']
    #    if len(ref['answers']) > 0:
    #        pred_dict[question_id] = normalize(pred['answers'])
    #        ref_dict[question_id] = normalize(ref['answers'])
    #    bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)

    #reader = reader_factory(reader_config_path)
    


def test_dureader_bidaf_rc(test_path,reader_exp_name,para_selection_method):
    print('test_dureader_bert_rc loading samples...')
    #
    #sample_list = loader.sample_list

    
    reader = ReaderFactory.from_exp_name(reader_exp_name,READER_CLASS='bidaf')
    #reader_config = reader.config
    #loader = DureaderLoader(test_path,para_selection_method,sample_fields=['question','answers','question_id','question_type'])
    while True:
        print('???')
    #dataset  = BertRCDataset(losample_list,reader_config.MAX_QUERY_LEN,reader_config.MAX_SEQ_LEN,device=reader.device)
    #print('make batch')
    #iterator = dataset.make_batchiter(batch_size=128)
    #_preds = reader.evaluate_on_batch(iterator)
    #_preds = group_dict_list(_preds,'question_id')
    #pred_answers  = MaxAllJudger().judge(_preds)

    #evaluate_mrc_bert(pred_answers)


    #answer_dict = group_dict_list(sample_list,'question_id', lambda x: {'answers':normalize([x[0]['answers']])})
    
    #print(answer_dict)
    #print(pred_answers)
    #evaluate_mrc_bidaf(pred_answers)


def test_bleu_rouge():
    bleu_rouge = compute_bleu_rouge({'aaa':['你好嗎'],'bbb':['澳斑馬']}, {'aaa':['你好嗎真的好嗎','不好啦'],'bbb':['澳洲','斑馬']})
    print(bleu_rouge)



DUREADER_DEV_ALL = ['./data/devset/search.dev.json','./data/devset/zhidao.dev.json']
DEBUG_FILE = ['./data/demo/devset/search.dev.2.json']
#evaluate_chiu_rank('pointwise/answer_doc')
#test_dureader_bidaf_rc(DUREADER_DEV_ALL,'reader/bidaf','most_related_para')

#test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',{'class':'bert_ranker','kwargs':{'ranker_name':'pointwise/answer_doc'}})
#test_dureader_bert_rc(DUREADER_DEV_ALL,'reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})
test_dureader_bert_rc(DEBUG_FILE,'reader/bert_default',para_selection_method='most_related_para',decoder_dict={'class':'default','kwargs':{'k':1}})



