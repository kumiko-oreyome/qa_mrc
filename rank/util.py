import functools,json
from bert.modeling import BertForSequenceClassification
from bert.tokenization import BertTokenizer
from common.util import get_default_device,group_tuples
import  torch




    
# exmaples [tuple(Q,P)] 
# prediction [ score of Q,P [score 0,score 1]]
# return {q:[(pas,pred,label)]}
def aggregate_prediction(examples,predictions,labels=None,sort=False):
    questions,passages = tuple(zip(*examples))
    if labels is not None:
        l = [ (q,pas,pred,label)  for q,pas,pred,label in zip(questions,passages,predictions,labels)] 
    else:
         l = [ (q,pas,pred)  for q,pas,pred in zip(questions,passages,predictions)]
    group_dict = group_tuples(l,0)
    result = group_dict
    if sort:
        result = sort_preidction_by_score(group_dict)
    return result

# prediciton dict : {question:[(pas,pred),(pas,pred)....]}
def sort_preidction_by_score(preidciton_dict,attach_score=False,label_pos=1):
    result = {}
    for q,v in preidciton_dict.items():
        scores = [t[1][label_pos] for t in v]
        if attach_score:
            scores,payloads = sort_single_prediction_by_score(scores,[  (q,t[0],t[1][label_pos])   for t in v])
            assert isinstance(payloads,list)
        else:
           scores,payloads = sort_single_prediction_by_score(scores,v)
        result[q] = payloads
    return result


def sort_single_prediction_by_score(scores,payloads):
    assert isinstance(payloads,list) , 'type is %s'%(type(payloads))
    l = zip(scores,payloads)
    scores,payloads = tuple(zip(*sorted(l,key=lambda x:x[0],reverse=True)))
    return scores,list(payloads)




def predict_on_batch(model,dataloader,sigmoid=False):
    with torch.no_grad():
        _preds = []
        example_num = 0
        residual_num = 0
        for i,batch_X in enumerate(dataloader):
            if (residual_num) > 5000:
                residual_num-=5000
                #print('ranker predict on  %d th example ....'%(example_num))
            match_scores = model(batch_X[0],batch_X[1],batch_X[2])
            if sigmoid:
                match_scores  = torch.nn.Sigmoid()(match_scores)
            if isinstance(match_scores,list):
                _preds.extend(match_scores)
                continue
            if match_scores.is_cuda:
                _preds.extend([arr for arr in match_scores.cpu().numpy().tolist()])
            else:
                _preds.extend([arr for arr in match_scores.numpy().tolist()])
            example_num+=len(batch_X)
            residual_num = len(batch_X)
    return _preds



def model_factory(bert_path,device=None,tokenizer=None,**kwargs):
    if device is None:
        device = get_default_device()
    if tokenizer is None:
        tokenizer = BertTokenizer('%s/vocab.txt'%(bert_path))
    model = BertForSequenceClassification.from_pretrained(bert_path,num_labels=2,**kwargs).to(device)

    return model,tokenizer,device
