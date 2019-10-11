import functools
import torch
from . import util
from .datautil  import load_examples_from_scratch,generate_bert_pointwise_input,BatchIter,numeralize_fucntion_factory,evaluate_func_factory,sample_strategy_factory,\
     evaluate_on_examples,accuracy,precision
from common.experiment import Experiment,print_to_tee
import argparse
from dataloader.dureader import DureaderLoader



parser = argparse.ArgumentParser(description="train script")
parser.add_argument('exp_name')
parser.add_argument('--para_selection', default=None, nargs='?', choices=['most_related_para'])
parser.add_argument('--sample_method', default=None, nargs='?', choices=['answer_doc'])
parser.add_argument('--neg_k', default=9,type=int) 
parser.add_argument('--label_policy', default='answer_docs', const='answer_docs', nargs='?', choices=['answer_docs'])
#parser.add_argument('--eval_file', default='answer_docs', const='answer_docs', nargs='?', choices=['answer_docs'])                                                                      
args = parser.parse_args()  

EVAL_FILE = './data/devset/search.dev.json' 
#EVAL_FILE = './data/demo/devset/search.dev.json' 

expe =  Experiment(args.exp_name)
config = expe.config

output_file = expe.get_predict_file(prefix='eval',EVAL_FILE =EVAL_FILE,SAMPLE_MOTHOD=args.sample_method,NEG_K=args.neg_k,LABEL_POLICY=args.label_policy)
print_to_tee(output_file)

BATCH_SIZE = 128
LABEL_POS = 1
USE_LABEL = True
DEVICE =  util.get_default_device()



print('build model')
model,tokenizer,device  = util.model_factory(config.BERT_SERIALIZATION_DIR,device=DEVICE)
model.load_state_dict(torch.load(expe.model_path,map_location=device))   
model.eval()
_num_fn = numeralize_fucntion_factory(config.NUM_FN_NAME)
num_fn = functools.partial(_num_fn,max_seq_len=config.MAX_SEQ_LEN ,max_passage_len=config.MAX_PASSAGE_LEN ,tokenizer=tokenizer,device=device)


if args.para_selection == 'most_related_para' and  args.label_policy == 'answer_docs':
    loader = DureaderLoader(EVAL_FILE,args.para_selection,sample_fields=['question','answer_docs','question_id','question_type'])
    sample_list  = loader.sample_list
    sample_list  = list(filter(lambda x: len(x['answer_docs'])!=0,sample_list))
    #print(sample_list[0])
    X =  list(map(lambda x: (x['question'],x['passage']),sample_list))
    y =  list(map(lambda x: 1 if x['answer_docs'][0] == x['doc_id'] else 0 ,sample_list))
    print('total %d'%(len(X)))
    with torch.no_grad():
        metrics = [('accuracy',accuracy),('precision',precision)]
        results = evaluate_on_examples(model,num_fn,X,y,metrics,batch_size=128)
        print('- - - - - - ')
        print('metrics')
        for name,value in results.items():
            print('%s-->%.3f'%(name,value))
  


else:
    if args.sample_method is not None:
        sample_method = sample_strategy_factory(args.sample_method,k=args.neg_k)
    else:
        sample_method = None
        print('not using sample method !!!')

    print('load examples')
    eval_func = evaluate_func_factory(EVAL_FILE,num_fn,sample_stg=sample_method,label_policy=args.label_policy,batch_size=BATCH_SIZE)
    print('evaluating')
    results = eval_func(model)
