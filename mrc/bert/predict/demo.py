from bert.modeling import BertForQuestionAnswering, BertConfig
import torch
from bert.tokenization import BertTokenizer
from . import args
from  ..util import load_bert_rc_model

def read_from_demo_txt_file(path='../data/examples.txt'):

    def read_until(f,symbol):
        line = read_without_nl(f)
        if line is None:
            return None 
        if line == symbol:
            return '' 
        res = line+'\n'
        while True:
            line = read_without_nl(f)
            assert line is not None
            if line == symbol:
                res = res[:-1]
                break
            res+=line+'\n'
        return res

    def read_without_nl(f):
        l = f.readline()
        if len(l)>0  :
            l = l.rstrip()
        else:
            return None
        return l

    def readline_with_assert(f,symbol):
        line = read_without_nl(f)
        assert line is not None and symbol == line
        return line

    l = []
    with open(path,"r",encoding='utf-8') as f:
        while True:
            q = read_until(f,'>>>')
            if q is None:
                break
            passage = read_until(f,'<<<')
            assert passage is not None
            l.append((q,passage))
    return l






def find_best_span_from_probs(start_probs, end_probs,policy='greedy'):
        def greedy():
            best_start, best_end, max_prob = -1, -1, 0
            prob_start, best_start = torch.max(start_probs, 1)
            prob_end, best_end = torch.max(end_probs, 1)
            num = 0

            while True:
                if num > 3:
                    break
                if best_end >= best_start:
                    break
                else:
                    start_probs[0][best_start], end_probs[0][best_end] = 0.0, 0.0 #寫得很髒....
                    prob_start, best_start = torch.max(start_probs, 1)
                    prob_end, best_end = torch.max(end_probs, 1)
                num += 1
            max_prob = prob_start * prob_end

            if best_start <= best_end:
                return (best_start, best_end), max_prob
            else:
                return (best_end, best_start), max_prob

        def dp():
            sb,eb = start_probs[0],end_probs[0]
            passage_len = len(sb)
            best_start, best_end, max_prob = -1, -1, 0
            for start_idx in range(passage_len):
                for ans_len in range(passage_len):
                    end_idx = start_idx + ans_len
                    if end_idx >= passage_len:
                        continue
                    prob = sb[start_idx]+eb[end_idx]
                    if prob > max_prob:
                        best_start = start_idx
                        best_end = end_idx
                        max_prob = prob
            return (best_start, best_end), max_prob
          
        if policy == 'greedy':
            return greedy()
        else:
            return dp()




def extact_answer_from_span(q,p,span):
    text = "$" + q + "\n" + p
    answer = text[span[0]:span[1]+1]
    return answer

#retrun dict
def predict_one_sample(bert_model,sample_dict):
    start_prob,end_prob = bert_model(sample_dict["input"], sample_dict["seg"] ,attention_mask=sample_dict["att_mask"])
    span,prob = find_best_span_from_probs(start_prob,end_prob,policy='dp')
    answer = extact_answer_from_span(sample_dict['question'],sample['passage'],span)
    return answer

class BertInputConverter():
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    # return dict
    def convert(self,question,passage,max_q_len,max_seq_len,to_tensor=True):
        query_tokens = list(question)
        if len(query_tokens) > max_q_len:
            query_tokens = query_tokens[0:max_q_len]
        tokens, segment_ids = [], []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for i in list(passage):
            tokens.append(i)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        if len(tokens) > max_seq_len:
            tokens[max_seq_len-1] = "[SEP]"
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens[:max_seq_len])      ## !!! SEP
            segment_ids = segment_ids[:max_seq_len]
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)
        if not to_tensor:
            return {'question':question,'passage':passage,'input':torch.LongTensor(input_ids),'seg':torch.LongTensor(segment_ids),'att_mask':torch.LongTensor(input_mask)}
        return {'question':question,'passage':passage,'input':torch.LongTensor(input_ids).unsqueeze(0),'seg':torch.LongTensor(segment_ids).unsqueeze(0),'att_mask':torch.LongTensor(input_mask).unsqueeze(0)}

if __name__ == "__main__":
    print('load model')
    model = load_bert_model()
    print('convert')
    tokenizer =  BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    cvt =  BertInputConverter(tokenizer)
    examples = read_from_demo_txt_file()

    for question,passage in examples:
        sample = cvt.convert(question,passage,args.max_query_length,args.max_seq_length)
        print('Question')
        print(sample['question'])
        print('Passage')
        print(sample['passage'])
        answer = predict_one_sample(model,sample)
        print('Answer')
        print(answer)

