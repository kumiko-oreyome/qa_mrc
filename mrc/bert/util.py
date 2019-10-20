
from bert.modeling import BertForQuestionAnswering, BertConfig
import torch


def  load_bert_rc_model(config_path,wieght_path,device=None):
    config = BertConfig(config_path)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(wieght_path,map_location=device))
    if device is not None:
        return model.to(device)
    return model.cpu()


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
        # for pos map
        inc_n = len(tokens)
        pos_map = []
        for i,token in enumerate(list(passage)):
            tokens.append(token)
            segment_ids.append(1)
            pos_map.append(inc_n+i)
        tokens.append("[SEP]")
        segment_ids.append(1)
        if len(tokens) > max_seq_len:
            tokens[max_seq_len-1] = "[SEP]"
            pos_map = pos_map[0:max_seq_len-1]+[max_seq_len-2  for i in range( max_seq_len-1,len(pos_map))]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens[:max_seq_len])      ## !!! SEP
            segment_ids = segment_ids[:max_seq_len]
        else:
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) == len(segment_ids)
        if not to_tensor:
            return {'question':question,'passage':passage,'input':torch.LongTensor(input_ids),'seg':torch.LongTensor(segment_ids),'att_mask':torch.LongTensor(input_mask),'pos_map':pos_map}
        return {'question':question,'passage':passage,'input':torch.LongTensor(input_ids).unsqueeze(0),'seg':torch.LongTensor(segment_ids).unsqueeze(0),'att_mask':torch.LongTensor(input_mask).unsqueeze(0),'pos_map':pos_map}