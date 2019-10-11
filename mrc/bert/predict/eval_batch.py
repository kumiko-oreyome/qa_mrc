from torchtext.data import Dataset,Example,RawField,Iterator,Field
from bert.tokenization import BertTokenizer
import json
from . import args,predict_data
from tqdm import tqdm
from .demo import find_best_span_from_probs,extact_answer_from_span
import torch
from ..util import load_bert_rc_model,BertInputConverter
from ..metric import mrc_eval
from dataloader.dureader import DureaderLoader,BertRCDataset,BertRankDataset
from qa.reader import ReaderFactory
#qwqrqwrqwrsafsafsgsagas
#asgsagsgsgs
def process_batch(batch):
    l = []
    for arr in batch:
        l.append(arr)
    return to_tensor(l)
## TODO device argument sepcify by user
def to_tensor(arr,device=None,dtype=torch.long):
    if device is None:
        device = args.device   
    return torch.tensor(arr,dtype=dtype, device=device)

field1 = RawField()
field2 = Field(batch_first=True, sequential=True, tokenize=lambda ids:[int(i) for i in ids],use_vocab=False, pad_token=0)
FIELDS = [('question_id',field1),('question_type',field1),('passage',field1),('question_text',field1),('answers',field1),\
         ('input_ids',field2),('input_mask',field2),('segment_ids',field2)]

# a question will generate several examples of ##passage## and question examples
def make_examples(source,cvt):
    assert len(source['documents']) > 0
    sq = source['question'].strip()
    passage_list = []
    for doc in source['documents']:
        #ques_len = len(doc['segmented_title']) + 1
        #clean_passage = "".join(doc['segmented_paragraphs'][doc['most_related_para']][ques_len:])
        #if len(clean_passage) > 4:
        #    passage_list.append(clean_passage)
        passage_list.append(doc['paragraphs'][doc['most_related_para']])
    ret = []
    for passage in passage_list[:args.max_para_num]:
        sample =  cvt.convert(sq,passage, args.max_query_length, args.max_seq_length,to_tensor=False)
        (input_ids, input_mask, segment_ids) = sample['input'],sample['att_mask'],sample['seg']
        example = {'question_id':source['question_id'],
                'question_text':sq,
                'question_type':source['question_type'],
                'passage':passage,
                'answers':source['answers'],
                'input_ids':input_ids,
                'input_mask':input_mask,
                'segment_ids':segment_ids
        }
        ret.append(Example.fromdict(example,{t[0]:t for t in FIELDS}))
    return ret




def make_dataset(path_list):
    tokenizer =  BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    cvt = BertInputConverter(tokenizer)
    def process_file(path):
        l = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                l.extend(make_examples(json.loads(line.strip()),cvt))
        return l
    examples = []
    for path in path_list:
        examples.extend(process_file(path))
    dataset = Dataset(examples,FIELDS)
    return dataset



def make_batch_iterator(dataset,bs=32):
    return Iterator(dataset,batch_size=bs,train=False,sort_key=lambda x: len(x.input_ids),sort_within_batch=True,device=args.device)


def evaluate(evaluate_files,bert_config_path,weight_path,metric_dir,eval_method='bidaf_script'):
    print('load model')
    with torch.no_grad():
        model =  load_bert_rc_model(bert_config_path,weight_path,args.device)
        model = model.eval()
        dataset = make_dataset(evaluate_files)
        iterator = make_batch_iterator(dataset,bs=128)
        print('Iterate Batch')
        preds = []
        for  i,batch in enumerate(iterator):
            if i % 20 == 0:
                print('evaluate on %d batch'%(i))
            start_probs, end_probs = model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask)
            for i in range(len(start_probs)):
                sb,eb = start_probs[i].unsqueeze(0), end_probs[i].unsqueeze(0)
                span,score = find_best_span_from_probs(sb,eb)
                score = score.item() #輸出的score不是機率 所以不會介於0~1之間
                answer = extact_answer_from_span(batch.question_text[i],batch.passage[i],span)
                preds.append({'question_id':batch.question_id[i],'question':batch.question_text[i],'question_type': batch.question_type[i],
                                'answers': [answer],'entity_answers': [[]],'yesno_answers': [],'score':score,'gold':batch.answers[i]})

        tmp = {}
        for pred in preds:
            qid = pred['question_id']
            if qid not in tmp:
                tmp[qid] =[]
            tmp[qid].append(pred)

        pred_result,ref_result ={},{}
        # find max score predcition(dict) of qid 

        for qid in tmp:
            l = tmp[qid]
            max_answer = max(l,key=lambda d:d['score'])
            pred_result[qid] = max_answer

            ref = {k:v for k,v in max_answer.items()}
            ref['answers'] = max_answer['gold']
            ref_result[qid] = ref
        
        mrc_eval.evaluate(pred_result,ref_result)
     




def evaluate2(evaluate_files,bert_config_path,weight_path,metric_dir,eval_method='bidaf_script'):
    print('load model')
    with torch.no_grad():
        model =  ReaderFactory.from_exp_name('reader/bert_default',READER_CLASS='bert_reader').model
        model = model.eval()
        #dataset = make_dataset(evaluate_files)
        #iterator = make_batch_iterator(dataset,bs=128)
        loader = DureaderLoader(evaluate_files,'most_related_para',sample_fields=['question','answers','question_id','question_type'])
 
        dataset  = BertRCDataset(loader.sample_list, args.max_query_length, args.max_seq_length,device=args.device)
        iterator = dataset.make_batchiter(batch_size=128)
        print('Iterate Batch')
        preds = []
        for  i,batch in enumerate(iterator):
            if i % 20 == 0:
                print('evaluate on %d batch'%(i))
            start_probs, end_probs = model( batch.input_ids, token_type_ids= batch.segment_ids, attention_mask= batch.input_mask)
            for i in range(len(start_probs)):
                sb,eb = start_probs[i].unsqueeze(0), end_probs[i].unsqueeze(0)
                span,score = find_best_span_from_probs(sb,eb)
                score = score.item() #輸出的score不是機率 所以不會介於0~1之間
                answer = extact_answer_from_span(batch.question[i],batch.passage[i],span)
                preds.append({'question_id':batch.question_id[i],'question':batch.question[i],'question_type': batch.question_type[i],
                                'answers': [answer],'entity_answers': [[]],'yesno_answers': [],'score':score,'gold':batch.answers[i]})

        tmp = {}
        for pred in preds:
            qid = pred['question_id']
            if qid not in tmp:
                tmp[qid] =[]
            tmp[qid].append(pred)

        pred_result,ref_result ={},{}
        # find max score predcition(dict) of qid 

        for qid in tmp:
            l = tmp[qid]
            max_answer = max(l,key=lambda d:d['score'])
            pred_result[qid] = max_answer

            ref = {k:v for k,v in max_answer.items()}
            ref['answers'] = max_answer['gold']
            ref_result[qid] = ref
        
        mrc_eval.evaluate(pred_result,ref_result)




def evaluate3(evaluate_files,bert_config_path,weight_path,metric_dir,eval_method='bidaf_script'):
    from common.util import   group_dict_list
    print('load model')
    with torch.no_grad():
        reader =  ReaderFactory.from_exp_name('reader/bert_default',READER_CLASS='bert_reader')
        #dataset = make_dataset(evaluate_files)
        #iterator = make_batch_iterator(dataset,bs=128)
        loader = DureaderLoader(evaluate_files,'most_related_para',sample_fields=['question','answers','question_id','question_type'])
 
        dataset  = BertRCDataset(loader.sample_list, args.max_query_length, args.max_seq_length,device=args.device)
        iterator = dataset.make_batchiter(batch_size=128)
        print('Iterate Batch')
        preds = reader.evaluate_on_batch(iterator)

        tmp = {}
        tmp = group_dict_list(preds,'question_id')

        pred_result,ref_result ={},{}
        # find max score predcition(dict) of qid 

        for qid in tmp:
            l = tmp[qid]
            max_answer = max(l,key=lambda d:d['span_score'])
            max_answer.update({'entity_answers': [[]],'yesno_answers': []})
            ref = {k:v for k,v in max_answer.items()}
            ref_result[qid] = ref
            #順序不能倒過來...

            max_answer['answers'] = [ max_answer['span']]
            pred_result[qid] = max_answer
            

        
        mrc_eval.evaluate(pred_result,ref_result)

evaluate3(['./data/demo/devset/search.dev.json'],'./pretrained/chinese_wwm_ext_pytorch/bert_config.json','./experiments/reader/bert_default/model/model.bin','./mrc/metric/')

#evaluate(['./data/demo/devset/search.dev.json'],'./pretrained/chinese_wwm_ext_pytorch/bert_config.json','./experiments/reader/bert_default/model/model.bin','./mrc/metric/')
#evaluate(['./data/devset/search.dev.json'],'./pretrained/chinese_wwm_ext_pytorch/bert_config.json','./experiments/reader/bert_default/model/model.bin','./mrc/metric/')

