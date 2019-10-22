from rank.datautil import test_1,test_2,test_group_tuples,test_evaluate_on_file
from .reinforce import ReinforceBatchIter
#test_3()

#test_evaluate_on_file()
def test_reinforce_batch_iter():
    l = [{'k':123,'a':1},{'k':123,'a':2},{'k':456,'a':3},{'k':789,'a':4},{'k':111,'a':5},{'k':111,'a':6},{'k':222,'a':7},{'k':333,'a':8},{'k':10000,'a':9}]
    iterator = ReinforceBatchIter(l).get_batchiter(2,'k')
    for batch in iterator:
        print('- - - - -- ')
        print(batch)


class FakeTokenizer():
    def __init__(self):
        pass
    def convert_tokens_to_ids(self,tokens):
        return [0 for _ in range(len(tokens))]

def test_bert_input_convert():
    print('test_bert_input_convert')
    from .util import  BertInputConverter
    s = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXTZ~!=[]aabbccddeefffdokghoighoiehfpeogupihdslkgdshg'
    max_q_len = 10
    max_seq_len = 64
    cvt = BertInputConverter( FakeTokenizer())
    print('')
    for q_len in range(0,max_q_len+10):    
        print(q_len)
        for p_len in range(0,90):
            question = s[:q_len]
            passage = s[:p_len]
            for start in range(p_len):
                for end in range(start,p_len):
                    tmp = cvt.convert(question,passage,max_q_len, max_seq_len,to_tensor=False)
                    (input_ids, input_mask, segment_ids) = tmp['input'],tmp['att_mask'], tmp['seg']
                    a,b = tmp['pos_map'][start],tmp['pos_map'][end]
                    assert a>0 and b<len(input_ids)



def test_preprocessing_charspan():
    from dataloader.dureader import DureaderLoader
    from .util import preprocessing_charspan
    loader = DureaderLoader("./data/demo/devset/search.dev.2.json" ,'most_related_para',sample_fields=['question','answers','question_id','question_type','answer_docs','answer_spans'],\
        doc_fields=['segmented_paragraphs'])
    #print(len(loader.sample_list))
    #print(loader.sample_list[1])
    for sample in loader.sample_list:
        word_tokens = sample['segmented_paragraphs']
        preprocessing_charspan(sample)
        passage = sample['passage']
        start,end = sample['char_spans'][0]
        assert passage[start:end+1]  in "".join(word_tokens)
    print(loader.sample_list[3])





#def test_bert_tokenize():
#    from bert.tokenization import  BertTokenizer
#    tokenizer =  BertTokenizer('./pretrained/chinese_wwm_ext_pytorch/vocab.txt', do_lower_case=True)
#     tokenizer
    
test_preprocessing_charspan()