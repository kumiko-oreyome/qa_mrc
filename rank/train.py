import os
import functools
import torch

from torch.optim import SGD
from .datautil import  sample_strategy_factory,load_examples_from_scratch,evaluate_func_factory,BatchIter,generate_bert_pointwise_input,\
    numeralize_fucntion_factory
from .util import model_factory
from experiment import Experiment,print_to_tee
import argparse


# 樣本不平衡的情況看accuracy很不好
#用precision好了
def train_pointwise(model,train_batch,evaluate_func,epoch=10,optimizer=None,print_every=None,model_path=None):
    if optimizer is  None:
        optimizer = SGD(model.parameters(), lr=0.00001, momentum=0.9)
    if print_every  is None:
        print_every = 100

    best_performance = -1
    for i in range(epoch):
        loss_sum = 0
        iteration_cnt = 0
        print('Epoch : %d'%(i))
        for j,(batch_X,batch_y) in enumerate(train_batch):
            inputs,seg_ids,attn_masks= batch_X
            loss = model(inputs,seg_ids,attn_masks,batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum+=loss.item()
            iteration_cnt+=1
            if j % print_every == 0:
                print('iteration %d in epoch %d , loss is %.3f'%(j,i,loss.item()))
        print('loss of epoch %d is %.3f'%(i,loss_sum/iteration_cnt))
        performance = evaluate_func(model)
        if performance > best_performance:
            best_performance = performance
            print('best performance is %.3f'%(performance))
            torch.save(model.state_dict(),model_path)
        model.train()







parser = argparse.ArgumentParser(description="train script")
parser.add_argument('exp_name')   
parser.add_argument('--debug', action='store_true', help='verbose mode')   
                                                                                   
args = parser.parse_args()  


if args.debug:
    print('warning ... excute in debug mode')
    expe =  Experiment('debug/pointwise/answer_doc')
else:
    expe = Experiment(args.exp_name)


config  = expe.config
print_to_tee(expe.train_log_path)

sample_stg = sample_strategy_factory(config.SAMPLE_STG_NAME,k=config.NEG_SAMPLE_NUM)


print('load examples')
train_examples = load_examples_from_scratch(config.TRAIN_PATH,sample_stg ,concat=True)
print('build model')


model,tokenizer,device  = model_factory(config.BERT_SERIALIZATION_DIR)


_num_fn = numeralize_fucntion_factory(config.NUM_FN_NAME)
num_fn = functools.partial(_num_fn,max_seq_len=config.MAX_SEQ_LEN ,max_passage_len=config.MAX_PASSAGE_LEN ,tokenizer=tokenizer,device=device)
train_iter = BatchIter(train_examples,config.BATCH_SIZE,num_fn)


if config.MAX_SEQ_LEN > 200:
    evaluate_func = evaluate_func_factory(config.DEV_PATH,num_fn,sample_stg,batch_size=16)
else:
    evaluate_func = evaluate_func_factory(config.DEV_PATH,num_fn,sample_stg,batch_size=32)

print('start training')
train_pointwise(model,train_iter,evaluate_func,epoch=config.EPOCH,print_every=100,model_path=expe.model_path)


print('tranining finish...')