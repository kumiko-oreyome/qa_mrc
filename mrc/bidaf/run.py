# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module prepares and runs the whole system.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import pickle
import argparse
import logging
from .dataset import BRCDataset
from .vocab import Vocab
from .rc_model import RCModel
import jieba as jb

from mrc.bidaf import vocab
sys.modules['vocab'] = vocab



class PrintHandler(logging.Handler):
    def __init__(self,formatter):
        logging.Handler.__init__(self)
        self.setFormatter(formatter)
        self.setLevel(logging.DEBUG)
        #self.f = open('print.txt','w',encoding='utf-8')
    def emit(self, record):
        print(self.format(record),flush=True)
        #self.f.write('%s\n'%(self.format(record)))
        #self.f.flush()



def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--demo', action='store_true',
                        help='demo')
    parser.add_argument('--gold_paragraph', action='store_true')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['../data/demo/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['../data/demo/testset/search.test.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("brc")
    logger.info('Checking the data files...')
    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Load BRCD DATASET')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files, args.test_files)
    logger.info('Building vocabulary...')
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)
    logger.info('Built vocabulary , start filter vocab')

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))

    logger.info('Assigning embeddings...')
    vocab.randomly_init_embeddings(args.embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.set_vocab(vocab)
    logger.info('Initialize the model...')
    rc_model = RCModel(vocab, args)
    logger.info('Training the model...')
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files,gold_paragraph=args.gold_paragraph)
    logger.info('Converting text into ids...')
    brc_data.set_vocab(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))


def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("brc")
    logger.info('Load vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    logger.info('Load dataset...')
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files,gold_paragraph=args.gold_paragraph)
    logger.info('Converting text into ids...')
    brc_data.set_vocab(vocab)
    logger.info('Restoring the model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size)
    #for b in test_batches:
    #    logger.info('batch')
    #    logger.info('{}'.format(b['passage_token_ids'][0]))
    #    logger.info('{}'.format(b['passage_length'][0]))
    #    for k in b:
    #        logger.info('{} -->{}'.format(k,len(b[k])))
    #    break
    rc_model.evaluate(test_batches,result_dir=args.result_dir, result_prefix='test.predicted')

def read_from_demo_txt_file(path='./data/demo/examples.txt'):

    def read_until(f,symbol):
        line = read_without_nl(f)
        if line is None:
            return None 
        if line == symbol:
            return '' 
        res = line+'\n'
        while True:
            line = read_without_nl(f)
            assert line is not None
            if line == symbol:
                res = res[:-1]
                break
            res+=line+'\n'
        return res

    def read_without_nl(f):
        l = f.readline()
        if len(l)>0  :
            l = l.rstrip()
        else:
            return None
        return l

    def readline_with_assert(f,symbol):
        line = read_without_nl(f)
        assert line is not None and symbol == line
        return line

    l = []
    with open(path,"r",encoding='utf-8') as f:
        while True:
            q = read_until(f,'>>>')
            if q is None:
                break
            passage = read_until(f,'<<<')
            assert passage is not None
            l.append((q,passage))
    return l

def demo(args):
    print('demo')
    from  .dataset import _caculate_maxlen_batch,_padding_batch, _padding_from_tokens
    logger = logging.getLogger("brc")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    logger.info('Restore model...')
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('start demo')
    #demo_questions = ["微信", "分享", "链接", "打开", "app"]
    #demo_passages = ["微信", "已经成为", "现代人", "生活", "中", "必不可少", "的", "一部分", ",", "下面", "我", "就", "教", "大家", "如何", "在", "微信", "浏览器", "中", "打开", "本地", "APP", "吧", "!"\
    #    ,"1", "将", "手机", "微信", "打开", "。", "2", "打开", "微信", "中", "的", "链接", "。", "3", "我们", "打开", "百度", "经验", "的", "链接", "，", "用", "微信", "浏览器", "进入", "网页", "以后", "，", "点击", "右", "上方", "如图", "符号", "。", "4", "进入", "选择", "页面", "，", "点击", "“", "在", "浏览器", "”", "打开", "。", "5", "出现", "手机", "已", "安装", "的", "本地", "APP", "浏览器", "，", "我们", "选择", "一", "个", "自己", "想", "用", "的", "本地", "浏览器", "点击", "下方", "“", "仅", "一次", "”", "或", "“", "总是", "”", "都", "可以", "打开", "打开", "本地", "APP", "浏览器", "。", "6", "此时", "，", "我们", "就", "已经", "在", "微信", "浏览器", "中将", "本地", "APP", "浏览器", "打开", "了", "。"]
    
    
    read_from_demo_txt_file()
    #demo_questions = ["你好嗎"]
    #demo_passages = ["我很好啊哥哥爸爸真偉大主席率領"]
    demo_questions,demo_passages= tuple(zip(*read_from_demo_txt_file()))
    demo_questions_tokens , demo_passages_tokens = [list(jb.cut(q,cut_all=False)) for q in demo_questions]\
    ,[list(jb.cut(p,cut_all=False)) for p in demo_passages]
    q_token_ids = [vocab.convert_to_ids(tokens) for tokens in demo_questions_tokens]
    passage_token_ids =[vocab.convert_to_ids(tokens) for tokens in demo_passages_tokens]
    logger.info('question token numbers {}'.format(len(q_token_ids)))
    logger.info('passage token numbers {}'.format(len(passage_token_ids)))

    from .dataset import get_pad_id
    pad_id = get_pad_id(vocab)
    padd_q,q_lens =  _padding_from_tokens(q_token_ids ,pad_id,args.max_q_len)
    padd_p,p_lens =  _padding_from_tokens(passage_token_ids ,pad_id,args.max_p_len)
    batch = { 'questions' : demo_questions,
              'passages' : demo_passages,
              'question_token_ids': padd_q,
              'question_length': q_lens,
              'passage_token_ids': padd_p,
              'passage_length': p_lens}
    rc_model.demo(batch,'./demo.txt')

def test_logger(logger):
    import time
    while True:
        logger.info('hasaki')
        time.sleep(2)
        logger.info('hakayo.')
        time.sleep(2)

def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("brc")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.info('Running with args : {}'.format(args))
    phldr = PrintHandler(formatter)
    logger.addHandler( phldr)
 

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)
    if args.demo:
        demo(args)


if __name__ == '__main__':
    run()
