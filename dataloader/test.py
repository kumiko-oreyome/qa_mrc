from .dureader import  DureaderRawExample,DureaderRawDocument,DureaderLoader
from common.util import jsonl_reader

json_obj = next(jsonl_reader('./data/unittest/dureader_fake.json'))



example = DureaderRawExample(json_obj)
flatten_samples = example.flatten()
assert len(flatten_samples) == 5
assert json_obj['question'] == flatten_samples[0]['question']
assert json_obj['question_id'] == flatten_samples[0]['question_id']


o1 = [{'passage': 'abcde', 'passage_id': 0, 'doc_id': 0, 'question': 'wtf is this?', 'question_id': 12345}]
o2 =  [{'passage': 'abcde', 'passage_id': 0, 'doc_id': 0, 'question': 'wtf is this?', 'question_id': 12345}, {'passage': 'ccc', 'passage_id': 2, 'doc_id': 1, 'question': 'wtf is this?', 'question_id': 12345}]
sl1 = DureaderLoader('./data/unittest/dureader_fake.json',paragraph_selection='answer_doc').sample_list 
sl2 = DureaderLoader('./data/unittest/dureader_fake.json',paragraph_selection='most_related_para').sample_list 
assert DureaderLoader([],paragraph_selection='answer_doc')._para_selection(json_obj)== o1
assert DureaderLoader([],paragraph_selection='most_related_para')._para_selection(json_obj)
assert sl1== o1,'%s'%(sl1)
assert sl2== o2,'%s'%(sl2)

