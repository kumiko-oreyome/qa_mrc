class MaxAllJudger():
    def __init__(self):
        pass
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            ret[q] = []
            max_score = -10000000
            for d in v :
                if d['span_score'] > max_score:
                    max_score = d['span_score']
                    ret[q] = [d]
        return ret


class TopKJudger():
    def __init__(self,k=1):
        self.k = k
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            l= sorted(v,key=lambda x: -1*x['span_score'])
            ret[q] = [ x for x in l[:self.k]]
        return ret



class LambdaJudger():
    def __init__(self,k,score_func):
        self.k = k
        self.score_func = score_func
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            ret[q] = []
            max_score = -100000
            for d in v :
                d['judger_score'] = self.score_func(d)
            l= sorted(v,key=lambda x: -1*x['judger_score'])
            ret[q] = [ x for x in l[:self.k]]
               
        return ret
