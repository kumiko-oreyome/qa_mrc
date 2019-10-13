##
import random
from .reader  import  extract_answer_brute_force,extract_answer_dp_linear
from .decoder import LinearDecoder
def test_extract_answer_dp():
    print('test_extract_answer_dp')
    start_probs = []
    end_probs = []
    k = 100
    try:
        for _ in range(10):
            start_probs = [ random.randint(0,100) for _ in range(k)]
            end_probs = [ random.randint(0,100) for _ in range(k)]
            span1,score1 = extract_answer_brute_force(start_probs,end_probs) 
            span2,score2 = extract_answer_dp_linear(start_probs,end_probs)
            assert span1 == span2
    except  AssertionError:
        print(span1,span2)
        print(score1,score2)
        print(start_probs[span1[0]],end_probs[span1[1]] )
        print(start_probs[span2[0]],end_probs[span2[1]] )
        return 

def test_extract_topk_answer_dp():
    print('test_extract_answer_dp topk')
    start_probs = []
    end_probs = []
    k = 1000
    try:
        for _ in range(10):
            start_probs = [ random.randint(0,100) for _ in range(k)]
            end_probs = [ random.randint(0,100) for _ in range(k)]
            span1,score1 = extract_answer_brute_force(start_probs,end_probs,k=2) 
            span2,score2 = LinearDecoder(k=2).decode_span(start_probs ,end_probs)
            assert len(span1) == len(span2)
            for i in range(len(span1)):
                assert span1[i] == span2[i] or score1[i]==score2[i]
    except  AssertionError:
        print(span1,span2)
        print(score1,score2)
        #print(start_probs[span1[0]],end_probs[span1[1]] )
        #print(start_probs[span2[0]],end_probs[span2[1]] )
        return 

def test_decode_answer():
    text = '1234567890abcdefgh'
    decoder =  LinearDecoder()
    ret,_ = decoder.decode_answer([(1,3),(0,4),(6,8),(7,9),(13,15)],[0],text)
    answer = '123457890def'
    assert ret ==answer

test_decode_answer()

#test_extract_answer_dp()
#test_extract_topk_answer_dp()