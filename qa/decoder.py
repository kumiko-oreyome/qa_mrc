from common.util  import Factory








class MrcDecoder():
    def __init__(self):
        pass
    def decode(self,start_probs,end_probs,text):
        span,score = self.decode_span(start_probs,end_probs)
        answer,max_score = self.decode_answer(span,score,text)
        return answer,max_score,span

    def decode_span(self,start_probs,end_probs):
        pass
    def decode_answer(self,span,score,text):
        pass

class LinearDecoder(MrcDecoder):
    def __init__(self,k=1):
        self.k = k
    def decode_span(self,start_probs,end_probs):
        top_k_spans = []
        N = len(start_probs)
        assert N>0
        top_k_spans.append((0,0,start_probs[0]+end_probs[0]))
        new_topk_span_candidates = [(0,start_probs[0])]
        for i in range(1,N):   
            new_topk_span_candidates.append((i,start_probs[i]))
            new_topk_span_candidates =  list(sorted(new_topk_span_candidates,key=lambda x: x[1],reverse=True))[0:self.k]
            top_k_spans.extend([ (si,i,start_probs[si]+end_probs[i]) for si,_ in new_topk_span_candidates])
            top_k_spans = list(sorted(top_k_spans,key=lambda x: x[2],reverse=True))[0:self.k]
        return list(map(lambda x: (x[0],x[1]),top_k_spans)),list(map(lambda x: x[2],top_k_spans))

    def decode_answer(self,span,score,text):
        intervals = []
        for start,end in span:
            intv_flag = False
            for i,(intv_start,intv_end) in enumerate(intervals):
                if max(intv_start,start) <= min(intv_end,end):
                    intv_flag = True
                    intervals[i] = ( min(intv_start,start),max(intv_end,end))
                    break
            if not intv_flag:
                intervals.append((start,end))
        # concat spans
        ret = ''
        for intv in intervals:
            ret+= text[intv[0]:intv[1]+1]

        return ret,max(score)
            






class MrcDecoderFactory(Factory):
    NAME2CLS = {"default":LinearDecoder}
    def __init__(self):
        pass
