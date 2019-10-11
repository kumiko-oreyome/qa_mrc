import functools

import torch

from . import util
from .datautil  import load_examples_from_scratch,generate_bert_pointwise_input,BatchIter,numeralize_fucntion_factory
from experiment import Experiment,print_to_tee
import argparse


parser = argparse.ArgumentParser(description="train script")
parser.add_argument('exp_name')   
parser.add_argument('file')
parser.add_argument('--debug', action='store_true', help='verbose mode')   
parser.add_argument('--label_policy',default='answer_docs')  
parser.add_argument('--without_label', action='store_true')  



args = parser.parse_args()  

if args.debug:
    print('warning ... excute in debug mode')
    expe =  Experiment('debug/pointwise/answer_doc')
else:
    expe = Experiment(args.exp_name)


DATA_SRC_FILE = args.file
LABEL_POLICY = args.label_policy

BATCH_SIZE = 16
LABEL_POS = 1
USE_LABEL = not args.without_label
DEVICE =  util.get_default_device()


config  = expe.config
output_file = expe.get_predict_file(prefix='predict',LABEL_POLICY=LABEL_POLICY,DATA_SRC_FILE=DATA_SRC_FILE)
print_to_tee(output_file)


print('load examples')
if USE_LABEL:
    examples,labels = load_examples_from_scratch(DATA_SRC_FILE,attach_label=LABEL_POLICY)
else:
    examples,labels = load_examples_from_scratch(DATA_SRC_FILE),None

print('build model')
model,tokenizer,device  = util.model_factory(config.BERT_SERIALIZATION_DIR,device=DEVICE)
model.load_state_dict(torch.load(expe.model_path,map_location=device))   
model.eval()

_num_fn = numeralize_fucntion_factory(config.NUM_FN_NAME)
num_fn = functools.partial(_num_fn,max_seq_len=config.MAX_SEQ_LEN ,max_passage_len=config.MAX_PASSAGE_LEN ,tokenizer=tokenizer,device=device)
batch_iter = BatchIter(examples,BATCH_SIZE,num_fn)

predictions = util.predict_on_batch(model,batch_iter)
print('prediction finish.  (total:%d predictions)'%(len(predictions)))
result = util.aggregate_prediction(examples,predictions,labels=labels,sort=True)

for q,v in result.items():
    print('Question :%s'%(q))
    print('- - - - - ')
    print('Passages:')
    for i,(pas,pred,*label) in enumerate(v[0:10]):
        print('rank %d'%(i+1))
        print(pred[LABEL_POS])
        if len(label)>0:
            label = label[0]
            print('label is %d'%(label))

        print(pas[0:100])
    print('- - - ')



