import unicodedata
import random
#問題與答案分類一
def read_qa_txt(path,record_type='tuple'):
    ret = []
    with open(path,'r',encoding='utf-8') as f:
        while True:
            title = f.readline().rstrip()
            print(title)
            if len(title)<1:
                break
            answer = f.readline().rstrip()
            title = unicodedata.normalize("NFKD", title).replace(" ", "") #去掉空白 /xa0
            answer = unicodedata.normalize("NFKD", answer).replace(" ", "")
            _ = f.readline()
            if record_type == 'tuple':
                ret.append((title,answer))
            elif record_type == 'dict': 
                ret.append({'title':title,'answer':answer})
            else:
                assert False
    return ret

def generate_pairwise_examples(examples,sample_k=None):
    assert len(examples) > 0
    ret = [] 
    for i,(q,_) in enumerate(examples):
        qa_list = []
        for j,(_,a) in enumerate(examples):
            label = 0 
            if i == j:
                label = 1
            qa_list.append((q,a,label))
        if sample_k is not None:
            neg_idx = [k for k in range(len(qa_list)) if k != i]
            neg_idx = random.sample(neg_idx,sample_k)
            qa_list = [qa_list[i]] +[  qa_list[k] for k in neg_idx]
        ret.extend(qa_list)


    return ret



