import os
from datetime import datetime
from common.util import load_json_config
import builtins
import functools
import io

def print_to_tee(filepath):
    if isinstance(filepath,str):
        f = open(filepath,'w',encoding='utf-8')
    elif isinstance(filepath,io.IOBase):
        f = filepath
    else:
        assert False
    print1 = functools.partial(builtins.print,flush=True)
    print2 = functools.partial(builtins.print,file=f,flush=True)
    def tee(s):
        print1(s)
        print2(s)
    builtins.print = tee


class Experiment():
    def __init__(self,exp_name,root_dir='./experiments'):
        self.exp_name = exp_name
        self.root_dir = root_dir
        self.exp_dir = '%s/%s'%(self.root_dir,self.exp_name)

        self.config_path = '%s/config.json'%(self.exp_dir)


        self.train_log_path = '%s/log.txt'%(self.exp_dir)

        self.model_dir = '%s/model'%(self.exp_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_path = '%s/model.bin'%(self.model_dir)
               
        self.predict_dir = '%s/predicts'%(self.exp_dir)
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir) 

        self.config = load_json_config(self.config_path,MODEL_PATH=self.model_path,PREDICT_DIR=self.predict_dir,MODEL_DIR=self.model_dir)
    



    def get_predict_file(self,path=None,prefix='',return_path=False,**hyper_paramters):
        if path is None:
            path  = '%s/%s%s.txt'%(self.predict_dir,prefix,datetime.now().strftime('%Y%m%d_%H%M%S'))
            print(path)
        if return_path:
            return path
        f = open(path,'w',encoding='utf-8')
        for k,v in hyper_paramters.items():
            f.write("%s:%s\n"%(k,v))
        return f

