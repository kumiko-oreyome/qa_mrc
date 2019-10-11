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


class MultiplyJudger():
    def __init__(self):
        pass
    def judge(self,documents):
        ret = {}
        for q,v in documents.items():
            ret[q] = []
            max_score = -100000
            for d in v :
                score = d['span_score']*d['rank_score']
                if score > max_score:
                    max_score =score
                    ret[q] = [d]
        return ret
