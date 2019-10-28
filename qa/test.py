import random
from .reader  import  extract_answer_brute_force,extract_answer_dp_linear
from .decoder import LinearDecoder

def test_extract_answer_dp():
    print('test_extract_answer_dp')
    start_probs = []
    end_probs = []
    k = 100
    try:
        for _ in range(10):
            start_probs = [ random.randint(-512,512) for _ in range(k)]
            end_probs = [ random.randint(-512,512) for _ in range(k)]
            span1,score1 = extract_answer_brute_force(start_probs,end_probs) 
            span2,score2 = extract_answer_dp_linear(start_probs,end_probs)
            assert span1[0] == span2 or score1[0]==score2
    except  AssertionError:
        print(span1,span2)
        print(score1,score2)
        print(start_probs[span1[0][0]],end_probs[span1[0][1]] )
        print(start_probs[span2[0]],end_probs[span2[1]] )
        return 

def test_extract_topk_answer_dp():
    print('test_extract_answer_dp topk')
    start_probs = []
    end_probs = []
    k = 100
    try:
        for _ in range(100):
            start_probs = [ random.randint(-512,512) for _ in range(k)]
            end_probs = [ random.randint(-512,512) for _ in range(k)]
            span1,score1 = extract_answer_brute_force(start_probs,end_probs,k=1) 
            span2,score2 = LinearDecoder(k=1).decode_span(start_probs ,end_probs)
            assert len(span1) == len(span2)
            for i in range(len(span1)):
                assert span1[i] == span2[i] or score1[i]==score2[i]
    except  AssertionError:
        print(span1,span2)
        print(score1,score2)
        #print(start_probs[span1[0]],end_probs[span1[1]] )
        #print(start_probs[span2[0]],end_probs[span2[1]] )
        return 

def test_decode_answer():
    text = '1234567890abcdefgh'
    decoder =  LinearDecoder()
    ret,_ = decoder.decode_answer([(1,3),(0,4),(6,8),(7,9),(13,15)],[0],text)
    print(ret)
    print(_)
    answer = '123457890def'
    assert ret ==answer
    
# integrate test
def test_mrc_baseline():
    print('test_dureader_bert_rc loading samples...')
    from dataloader.dureader import  DureaderLoader
    from qa.reader import ReaderFactory,BertRCDataset
    from qa.judger import MaxAllJudger
    from common.util import group_dict_list,evaluate_mrc_bidaf
    loader = DureaderLoader( ['./data/demo/devset/search.dev.2.json'],'most_related_para',sample_fields=['question','answers','question_id','question_type'])
    sample_list = loader.sample_list
    reader = ReaderFactory.from_exp_name('reader/bert_default',decoder_dict={'class':'default','kwargs':{'k':1}})
    reader_config = reader.config
    dataset  = BertRCDataset(sample_list,reader_config.MAX_QUERY_LEN,reader_config.MAX_SEQ_LEN,device=reader.device)
    print('make batch')
    iterator = dataset.make_batchiter(batch_size=128)
    _preds = reader.evaluate_on_batch(iterator)
    _preds = group_dict_list(_preds,'question_id')
    pred_answers  = MaxAllJudger().judge(_preds)
    res_dict = evaluate_mrc_bidaf(pred_answers)
    assert res_dict == {'Bleu-1': 0.19711538461443695, 'Bleu-2': 0.15154174071281326, 'Bleu-3': 0.11637351097094059, 'Bleu-4': 0.0983666932134996, 'Rouge-L': 0.260079879764384}
   
def test_bleu_rouge():
    from common.dureader_eval  import  compute_bleu_rouge,normalize
    bleu_rouge = compute_bleu_rouge({'aaa':['你好嗎'],'bbb':['澳斑馬']}, {'aaa':['你好嗎真的好嗎','不好啦'],'bbb':['澳洲','斑馬']})
    print(bleu_rouge)


def test_precision_recall():
    from qa.eval import precision,recall,accuracy
    rank_result = {'question1':[{'rank_score':0.86,'label':1},{'rank_score':0.22,'label':0},{'rank_score':0.92,'label':0}],\
                   'question2':[{'rank_score':0.4,'label':0},{'rank_score':0.5,'label':1},{'rank_score':0.1,'label':0},{'rank_score':0.2,'label':0}]}
    assert recall(rank_result,1) == precision(rank_result,1)
    assert recall(rank_result,1) ==0.5
    assert accuracy(rank_result) == (2/3+4/4)/2

test_extract_answer_dp()
test_extract_topk_answer_dp()
test_mrc_baseline()
test_precision_recall()