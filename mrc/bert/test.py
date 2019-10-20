from rank.datautil import test_1,test_2,test_group_tuples,test_evaluate_on_file
from .reinforce import ReinforceBatchIter
#test_3()

#test_evaluate_on_file()
def test_reinforce_batch_iter():
    l = [{'k':123,'a':1},{'k':123,'a':2},{'k':456,'a':3},{'k':789,'a':4},{'k':111,'a':5},{'k':111,'a':6},{'k':222,'a':7},{'k':333,'a':8},{'k':10000,'a':9}]
    iterator = ReinforceBatchIter(l).get_batchiter(2,'k')
    for batch in iterator:
        print('- - - - -- ')
        print(batch)

test_reinforce_batch_iter()
    
