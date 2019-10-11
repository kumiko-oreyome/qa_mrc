import itertools,json,random,functools
from operator import itemgetter
from bert.tokenization import BertTokenizer
from bert.modeling import BertForSequenceClassification
import torch
from . import util
from .util import group_tuples,sort_single_prediction_by_score
from common.util import jsonl_reader
##變化的部分
## example strategy : point_wise or pair_wise
## paragraph_selection strategy : document levels or sample levels
    ## match strategy
    ## trival stargey
    ## defined
# sample strategy XX


#class ExperimentConfig():
#    def __init__(self,*args,**dict_for_vars):
#        self.attr_dict = dict_for_vars
#        for k in dict_for_vars:
#            setattr((self,dict_for_vars[k]))
#POINTWISE_CONFIG = ExperimentConfig()    

def _fix_length_op(seq,min_len=None,max_len=None,pad_value=0):
    if max_len is not None and len(seq) > max_len:
        seq = seq[0:max_len]
    if min_len is not None and len(seq) < min_len:
        seq = seq + [pad_value]*(min_len-len(seq))
    return seq

def get_all_paragraphs_from_sample(sample):
    return list(itertools.chain( *[ doc["paragraphs"]  for doc in sample["documents"]]))

def wrap_tensor(arr,device):
    return torch.tensor(arr,device=device, dtype=torch.long)


        
def sample_template_factory(with_most_related_doc=False,with_most_related_para=False):
    if not with_most_related_para :
        DOC_TEMPLATE = {
            'paragraphs':'paragraphs'
        }
    else:
        DOC_TEMPLATE = {
            'paragraphs':'paragraphs','most_related_para':'most_related_para'
        }       
    if not with_most_related_doc:
        POINTWISE_TEMPLATE ={
            'question':'question',
            'documents':[DOC_TEMPLATE]
        }
    else:
        POINTWISE_TEMPLATE ={
            'question':'question',
            'documents':[DOC_TEMPLATE],
            'answer_docs':'answer_docs'
        }
    return POINTWISE_TEMPLATE


def extract_json_to_raw_data(json_obj,template):
    ret = {}
    for k,v in template.items():
        if isinstance(v,dict):
            extracted_obj = extract_json_to_raw_data(json_obj[k],template[k])
            field_name = k
        elif isinstance(v,list):
            extracted_obj = [ extract_json_to_raw_data(item,template[k][0]) for item in json_obj[k]]
            field_name = k
        else:
            assert isinstance(v,str)
            extracted_obj = json_obj[k]
            field_name = v
        ret[field_name] = extracted_obj

    return ret




def numeralize_fucntion_factory(name):
    table ={}
    table['pointwise'] = generate_bert_pointwise_input
    return table[name]


def sample_strategy_factory(name,**kwargs):
    stg_map = {'every_doc':sample_every_doc_by_label,'answer_doc':sample_answer_doc_by_label}
    assert name in stg_map
    stg = functools.partial(stg_map[name],**kwargs)
    return stg

def sample_every_doc_by_label(sample,k=1):
    pos_examples = []
    neg_examples = []
    for doc in sample["documents"]:
        if 'most_related_para' not in doc:
            continue        
        paras = doc['paragraphs']
        pos_idx = doc['most_related_para']
        pos_examples.append(paras[pos_idx])
        if len(paras)>1:
            neg_idx_list = [ idx for idx in range(len(paras)) if idx!=pos_idx]
            neg_num = min(k,len(neg_idx_list))
            neg_idxs = random.sample(neg_idx_list,neg_num)
            nl =[ paras[ni] for ni in neg_idxs]
            neg_examples.extend(nl)
    return pos_examples,neg_examples

def sample_answer_doc_by_label(sample,k=1):
    pos_examples = []
    neg_examples = []

    if 'answer_docs' not in sample or len(sample['answer_docs'])==0:
        return pos_examples,neg_examples
    answer_doc_idx = sample['answer_docs'][0]
    answer_doc = sample["documents"][answer_doc_idx]

    paras = answer_doc['paragraphs']

    pos_idx = answer_doc['most_related_para']
    pos_examples = [paras[pos_idx]]

    neg_idx_list = [ idx for idx in range(len(paras)) if idx!=pos_idx]
    neg_num = min(k,len(neg_idx_list))
    #assert neg_num>0
    if neg_num > 0:
        neg_idxs = random.sample(neg_idx_list,neg_num)
        neg_examples    = [ paras[ni] for ni in neg_idxs]

    return pos_examples,neg_examples   



def load_examples_from_scratch(path,sample_stg=None,concat=False,attach_label=None):
    examples = []
    labels = []
    line_cnt  = 0
    for json_obj in jsonl_reader(path):
        if line_cnt%2000 == 0:
            print('load %dth line'%(line_cnt))
        line_cnt+=1
        paras = []
        if sample_stg is None:
            tmp_paras = []
            tmp_labels = []
            for di,doc in enumerate(json_obj['documents']):
                if attach_label is not None and 'most_related_para' not in doc:
                    continue
                tmp_paras.extend(doc['paragraphs'])
                if attach_label is None:
                    continue
                zeros = [0] * len(doc['paragraphs'])
                if attach_label == 'most_related_para' :
                    zeros[doc['most_related_para']] = 1
                if attach_label == 'answer_docs' and 'answer_docs' in json_obj and  di in json_obj['answer_docs' ]:
                    zeros[doc['most_related_para']] = 1
                assert sum(zeros) <2
                tmp_labels.extend(zeros)
            if attach_label is not None:
                assert len(tmp_labels) == len(tmp_paras)
            paras.extend(tmp_paras)
            if len(tmp_labels) > 0:
                labels.extend(tmp_labels)
        else:
            pos_examples,neg_examples = sample_stg(json_obj)
            labels.extend([1]*len(pos_examples)+[0]*len(neg_examples))
            paras = pos_examples+neg_examples
        examples.extend([(json_obj['question'].strip(),p) for p in paras])
    print('total %d examples'%(len(examples)))
    if len(labels) > 0:
        if concat:
            return [(q,p,lb) for (q,p),lb in zip(examples,labels)  ] 
        return examples,labels
    return examples




def upwrap_batch(batch):
    assert len(batch) >0
    return tuple(zip(*batch))

def generate_bert_pointwise_input(examples,max_seq_len,max_passage_len,tokenizer,device,wrap_tensor_flag=True):
    y_flag = True
    if len(examples[0]) > 2:
        questions,passages,labels = list(zip(*examples))
    else:
        questions,passages = list(zip(*examples))
        y_flag = False
    X = []
    truncated_examples = []
    for question,passage in zip(questions,passages):
        q_tokens = tokenizer.tokenize(question)
        p_tokens = tokenizer.tokenize(passage)
        q_tokens = _fix_length_op(q_tokens,max_len=max_passage_len)
        p_tokens = _fix_length_op(p_tokens,max_len=max_passage_len)
        p_tokens = _fix_length_op(p_tokens,max_len=max_seq_len-3-len(q_tokens))
        #if len(p_tokens)+len(q_tokens) < max_seq_len-3:
        #    q_tokens = _fix_length_op(q_tokens,max_len=max_seq_len-3-len(p_tokens))
        assert len(p_tokens)+len(q_tokens) <= max_seq_len-3
        truncated_examples.append((q_tokens,p_tokens))
    padded_max_length = max([ len(a)+len(b)+3 for a,b in truncated_examples])  
    for q_tokens,p_tokens in truncated_examples:
        ql,pl = len(q_tokens),len(p_tokens)
        bert_input = tokenizer.convert_tokens_to_ids(["[CLS]"]+q_tokens+["[SEP]"]+p_tokens+["[SEP]"])
        seg_ids = [0]*(ql+2)+[1]*(pl+1)
        input_mask = [1]*len(bert_input)
        bert_input = _fix_length_op(bert_input,min_len=padded_max_length)
        seg_ids = _fix_length_op(seg_ids,min_len=padded_max_length)
        input_mask = _fix_length_op(input_mask,min_len=padded_max_length,pad_value=0)
        X.append((bert_input,seg_ids,input_mask))
    bert_input,seg_ids,input_mask = tuple(zip(*X))
    
    if not wrap_tensor_flag:
        if y_flag:
            return  (bert_input,seg_ids,input_mask),labels,device
        else:
            return  bert_input,seg_ids,input_mask

    t1,t2,t3 = wrap_tensor(bert_input,device),wrap_tensor(seg_ids,device),wrap_tensor(input_mask,device)
    if y_flag:
        return  (t1,t2,t3),wrap_tensor(labels,device)
    else:
         return  t1,t2,t3





class BatchIter():
    def __init__(self,examples,batch_size,numeralize_fn):
        self.numeralize_fn = numeralize_fn
        self.batch_size = batch_size
        self.examples = examples
        print('batch size is %d'%(self.batch_size))


    def __iter__(self):
        return self._gen_batch()

    def _gen_batch(self):
        cnt = 0
        while True:
            if cnt >= len(self.examples):
                break
            start_idx = cnt
            end_idx = start_idx+self.batch_size-1
            if end_idx>=len(self.examples):
                end_idx = len(self.examples)-1
            tensors = self.numeralize_fn(self.examples[start_idx:end_idx+1])
            cnt+= end_idx-start_idx+1
            yield tensors
     


def accuracy(prediction,labels):
    n = len(labels)
    argmax_prediction = [  l.index(max(l)) for l in prediction ]
    acc = sum([ 1 for  a,b in zip(argmax_prediction,labels) if a == b])
    return acc/n

def precision(match_scores,labels,k=1,label_pos=1):
    assert len(match_scores) == len(labels)
    if k > len(labels):
        k = len(labels)
    scores,payloads = [ v[label_pos] for v in match_scores ],list(labels)
    _,labels = sort_single_prediction_by_score(scores,payloads)
    label_k = labels[0:k]
    return sum(label_k)/k


def evaluate_func_factory(path,num_fn,sample_stg=None,**kwargs):
    def  evaluate_func(model):
        results = evaluate_on_file(path,model,num_fn,metrics=[('accuracy',accuracy),('precision',precision)],sample_stg=sample_stg,**kwargs)
        print('- - - - - - ')
        print('metrics')
        for name,value in results.items():
            print('%s-->%.3f'%(name,value))
        return results['precision']
    return evaluate_func

def evaluate_on_file(path,model,num_fn,metrics=[],sample_stg=None,batch_size=32,label_policy='answer_docs'):
    model.eval()
    with torch.no_grad():
        if sample_stg is  None:
            X,y = load_examples_from_scratch(path,concat=False,attach_label=label_policy)
        else: 
            X,y = load_examples_from_scratch(path,sample_stg=sample_stg,concat=False)
        results = evaluate_on_examples(model,num_fn,X,y,metrics,batch_size=batch_size)
    return results


def evaluate_on_examples(model,num_fn,X,y,metrics,batch_size=32):
    results = {}
    batchiter = BatchIter(X,batch_size,num_fn)
    for metric_name,metric in metrics:
        cnt = 0
        total = 0 
        _preds = util.predict_on_batch(model,batchiter)
        if  metric_name=='accuracy':
            group_dict =  {i:[(v,label)]  for i,(v,label) in enumerate(zip(_preds,y))}
        else :
            group_dict =  group_tuples( [(Q,v,label)    for (Q,_),v,label    in zip(X,_preds,y)],0)
        for _,gv in  group_dict.items():
            grouped_preds , grouped_labels  =  tuple(zip(*gv))
            value = metric(grouped_preds , grouped_labels)
            cnt+=1
            total+=value
        results[metric_name]=total/cnt
    return results


def test_evaluate_on_file():
    BERT_SERIALIZATION_DIR = './pretrained/chinese_wwm_ext_pytorch'  
    tokenizer = BertTokenizer('%s/vocab.txt'%(BERT_SERIALIZATION_DIR))
    device = torch.device('cpu')
    num_fn = functools.partial(generate_bert_pointwise_input,max_seq_len=200,max_passage_len=100,tokenizer=tokenizer,device=device)

    fake_model1 = lambda x,y,z:   [[0,1] for _ in range(len(x))]
    fake_model2 = lambda x,y,z:   [[1,0] for _ in range(len(x))]
    fake_model3 = lambda x,y,z:   [[random.choice([0,1]),random.choice([0,1])]for _ in range(len(x))]
    fake_model4 = lambda x,y,z:   [[random.uniform(0,1),random.choice([0,1])] for _ in range(len(x))]

    fake_model1.eval = lambda :None
    fake_model2.eval = lambda :None
    fake_model3.eval = lambda :None
    fake_model4.eval = lambda :None

    test_path = './data/demo/devset/search.dev.2.json'
    results1 = evaluate_on_file(test_path,fake_model1,num_fn,[('accuracy',accuracy),('precision',precision)])
    results2 = evaluate_on_file(test_path,fake_model2,num_fn,[('accuracy',accuracy),('precision',precision)])
    results3 = evaluate_on_file(test_path,fake_model3,num_fn,[('accuracy',accuracy),('precision',precision)])
    results4 = evaluate_on_file(test_path,fake_model4,num_fn,[('precision',precision),('precision2',functools.partial(precision,k=2))])
    X,y = load_examples_from_scratch(test_path,concat=False,attach_label='most_related_para')
    assert results1['accuracy'] == sum(y)/len(y)
    assert results2['accuracy'] == (len(y)-sum(y))/len(y)

    assert precision([[-1,1],[0,2],[0,3],[-1,-1]],[0,1,0,0],k=2)==0.5
    assert precision([[-1,1],[0,2],[0,3],[-1,-1]],[0,0,0,1],k=2)==0
    print(results3['accuracy'])
    print(results1['precision'])
    print(results2['precision'])
    print(results4['precision'])
    print(results4['precision2'])





    



def test_1():
    test_sample = {'question':'test question','answer_docs':[1],'aaas':1234,"documents": [{'paragraphs':['p1a','p1b','p1c'],'most_related_para':1,'span':[1,2]},{'paragraphs':['p2a','p2b'],'most_related_para':0,'span':[1,2]}]}
    template  = sample_template_factory(True,True)
    json_obj = extract_json_to_raw_data(test_sample,template)
    a,b = sample_every_doc_by_label(test_sample,1)
    c,d = sample_every_doc_by_label(test_sample,2)

    e,f = sample_answer_doc_by_label(json_obj)
    print(e)
    print(f)
    e,f = sample_answer_doc_by_label(json_obj,k=2)
    print(e)
    print(f)
    #print(get_all_paragraphs_from_sample(test_sample))
    #print(a)
    #print(b)
    #print(c)
    #print(d)

def test_2():
    datapath = './data/demo/devset/search.dev.json'
    stg = sample_strategy_factory('trivial_n',k=1)
    examples,labels = load_examples_from_scratch(datapath,stg)
    #examples,labels  = load_examples_from_scratch(datapath,attach_label='most_related_para')
    #for (q,p),label in zip(examples[0:20],labels[0:20]):
    #    print(q)
    #    print(p[0:50])
    #    print(label)
    #    print('##'*10)
    # 
    #print(len(examples))    
    #print(examples[0:10])
    #print(labels[0:10])

    examples = load_examples_from_scratch(datapath,None)
    #print(len(examples))
    examples = load_examples_from_scratch(datapath,stg,concat=True)

    BERT_SERIALIZATION_DIR = './pretrained/chinese_wwm_ext_pytorch'  
    tokenizer = BertTokenizer('%s/vocab.txt'%(BERT_SERIALIZATION_DIR))
    device = torch.device('cpu')
    
    num_fn = functools.partial(generate_bert_pointwise_input,max_seq_len=200,max_passage_len=100,tokenizer=tokenizer,device=device)
    
    fake_examples = [('你好嗎','歐巴馬撿到了300快'),('我不好啦','歐巴馬撿到了槍 高雄發大財了'),('哈哈哈','猜猜我是誰')]
    X = generate_bert_pointwise_input(fake_examples,20,7,tokenizer,device)
    for a,b,c in X:
        print('%d'%(a.shape))
        print('- - - '*18)
    #print(X)
    #print(examples[0:2])
    bt = BatchIter(examples,16,num_fn)
    for batch,y in bt:
        print(batch[0][0].shape)
        print(batch[1][1].shape)
        print(y.shape)



def test_group_tuples():
    data = [('aaa',1,'2'),('bb',0.5,3),('ggg','gg','g'),('aaa','gg',3)]
    print(group_tuples(data,0))
    print(group_tuples(data,1))
    print(group_tuples(data,0,True))
    print(group_tuples(data,1,True))

#test_evaluate_on_file()