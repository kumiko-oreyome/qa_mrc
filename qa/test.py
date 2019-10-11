##
import random
from .reader  import  extract_answer_brute_force,extract_answer_dp_linear
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


test_extract_answer_dp()