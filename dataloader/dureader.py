from common.util import jsonl_reader,get_default_device,group_dict_list
from mrc.bert.util import  BertInputConverter
from bert.tokenization import BertTokenizer
from torchtext.data import Dataset,Example,RawField,Iterator,Field
import functools
from rank.datautil import generate_bert_pointwise_input
from qa.para_select import ParagraphSelectorFactory

#DEFAULT_IGNORE_SAMPLE_FEILDS = ['answer_spans','answer_docs','fake_answers','answers','question_type']
#DEFAULT_IGNORE_DOC_FEILDS = ['title','most_related_para']



    




class DefaultSampleTransform():
    def __init__(self):
        pass

class DureaderRawExample():
    ALL_FIELDS = ['answer_spans','answer_docs','fake_answers','question','answers','question_id','question_type']
    def __init__(self,sample_obj):
        self.sample_obj = sample_obj

    def get_documents(self):
        return [ DureaderRawDocument(json) for json in self.sample_obj['documents']]

    def get_document_by_id(self,doc_id):
        return DureaderRawDocument(self.sample_obj['documents'][doc_id])

    def get_answer_doc(self):
        if len(self.sample_obj['answer_docs']) == 0:
            return None,None
        doc_id = self.sample_obj['answer_docs'][0]
        return self.get_document_by_id(doc_id),doc_id

    def flatten(self,sample_fields,doc_fields):
        ret = []
        for doc_id,doc in  enumerate(self.get_documents()):
            passage_list  = doc.flatten(doc_fields)
            for obj in passage_list:
                obj.update({ 'doc_id':doc_id}) 
                obj.update({ k:self.sample_obj[k] for k in sample_fields}) 
            ret.extend(passage_list)
        return ret


class DureaderRawDocument():
    ALL_FIELDS = ['title','most_related_para']
    def __init__(self,json_obj):
        self.json_obj = json_obj

    def get_most_related_para(self):
        para_id = self.json_obj['most_related_para']
        return self.get_paragraph_by_id(para_id),para_id

    def get_paragraphs(self):
        return self.json_obj['paragraphs']

    def get_paragraph_by_id(self,pid):
        return self.get_paragraphs()[pid]

    def flatten(self,fields=[]):
        ret = []
        for para_id,passage in enumerate(self.get_paragraphs()):
            obj = {'passage':passage,'passage_id':para_id}
            obj.update({ k:self.json_obj[k] for k in fields})
            if 'segmented_paragraphs' in fields:
                obj['segmented_paragraphs'] = obj['segmented_paragraphs'][para_id]
            ret.append(obj)
        return ret

    #def make_sample_dict(self,doc_id,para_id):
    #    retobj = { key:self.sample_obj[key] for key in self.sample_obj if key not in ['documents'] }
    #    para = self.get_document_by_id(doc_id).get_paragraph_by_id(para_id)
    #    retobj.update({'doc_id':doc_id,'passage_id':para_id,'passage':para})
    #    return retobj


class DureaderLoader():
    def __init__(self,path_list,paragraph_selection,sample_fields=['question','question_id'],doc_fields=[]):
        if isinstance(path_list,str):
            path_list = [path_list]
        self.path_list = path_list
        self.paragraph_selection = paragraph_selection
        self.sample_fields = sample_fields
        self.doc_fields = doc_fields
        self.sample_list = []
        if paragraph_selection is None or type(paragraph_selection)==str:
            self.paragraph_selector = paragraph_selection
        else:
            self.paragraph_selector = ParagraphSelectorFactory.create_selector(self.paragraph_selection)
        for path in path_list:
            print('dureader load %s'%(path))
            self.sample_list.extend(self._load_file(path))
        print('Dureader : total %d raw samples'%(len(self.sample_list)))


    def _load_file(self,path):
        ret = []
        for raw_sample in jsonl_reader(path):
            ret.extend(self._para_selection(raw_sample))
        return ret
        
    def _para_selection(self,raw_sample):
        drex = DureaderRawExample(raw_sample)
        sample_list = drex.flatten(self.sample_fields,self.doc_fields )

        if self.paragraph_selector is None:
            return sample_list
        if self.paragraph_selector in ['answer_doc','answer_docs']:
            answer_doc,ans_doc_id = drex.get_answer_doc()
            if ans_doc_id is None:
                answer_doc,ans_doc_id = drex.get_document_by_id(0),0
            answer_para,ans_para_id =  answer_doc.get_most_related_para()
            sample_list = list(filter(lambda dct:dct["doc_id"] ==ans_doc_id and dct["passage_id"] == ans_para_id,sample_list))
        elif self.paragraph_selector == 'most_related_para':
            legal_ids = [(doc_id,doc.get_most_related_para()[1]) for doc_id,doc in enumerate(drex.get_documents())]
            sample_list = list(filter(lambda dct:(dct["doc_id"],dct["passage_id"]) in legal_ids ,sample_list))
        elif type(self.paragraph_selector) == str:
            assert False
        else:
            sample_list = self.paragraph_selector.paragraph_selection(sample_list)

        return sample_list



RAW_FIELD = RawField()


class RecordDataset(object):
    #FIELD_MAP = {'passage':RAW_FIELD,'question':RAW_FIELD,'question_id':RAW_FIELD,'answers':RAW_FIELD,'question_type':RAW_FIELD}
    def __init__(self,sample_list,device=None):
        self.sample_list = sample_list
        self.device = get_default_device()
        self.fields = self.get_fields()

    def get_fields(self):
        assert len(self.sample_list[0].keys())>0
        fields = [(k,RAW_FIELD) for k in self.sample_list[0]]
        return fields




class BertRCDataset( RecordDataset):  
    bert_field = Field(batch_first=True, sequential=True, tokenize=lambda ids:[int(i) for i in ids],use_vocab=False, pad_token=0) 
    def __init__(self,sample_list,max_query_length,max_seq_length,train_flag=False,device=None):
        super(BertRCDataset,self).__init__(sample_list,device)
        self.max_query_length = max_query_length
        self.max_seq_length = max_seq_length
        self.tokenizer =  BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        self.cvt = BertInputConverter(self.tokenizer)
        self.train_flag = train_flag
        self.add_bert_fields()

        if train_flag:
            self.sample_list = [d for d in self.sample_list if len(d['char_spans'])==1]
        for sample in self.sample_list:
            tmp =  self.cvt.convert(sample['question'],sample['passage'],self.max_query_length, self.max_seq_length,to_tensor=False)
            (input_ids, input_mask, segment_ids) = tmp['input'],tmp['att_mask'], tmp['seg']
            sample.update({'input_ids':input_ids,'input_mask':input_mask,'segment_ids':segment_ids})
            if train_flag:
                ss,se =  sample['char_spans'][0] 
                sample['bert_span'] = tmp['pos_map'][ss],tmp['pos_map'][se]
                
    def add_bert_fields(self):
        self.fields+= [('input_ids',self.bert_field),('input_mask',self.bert_field),('segment_ids',self.bert_field)]
        if self.train_flag:
            self.fields.append(('bert_span',self.bert_field))

    def make_dataset(self):
        l = []
        for sample in  self.sample_list:
            l.append(Example.fromdict(sample,{t[0]:t for t in self.fields}))
        dataset = Dataset(l,self.fields)
        return dataset


    def make_batchiter(self,batch_size=32):
        dataset  = self.make_dataset()
        return Iterator(dataset,batch_size=batch_size,train=False,sort_key=lambda x: len(x.input_ids),sort_within_batch=True,device=self.device)


class BertRankDataset(RecordDataset):
    def __init__(self,sample_list,bert_path,max_passage_len,max_seq_length,device=None):
        super(BertRankDataset,self).__init__(sample_list,device)
        self.add_bert_fields()
        self.tokenizer = BertTokenizer('%s/vocab.txt'%(bert_path))
        self.max_seq_length = max_seq_length
        self.max_passage_len = max_passage_len
        #_num_fn = numeralize_fucntion_factory(config.NUM_FN_NAME)  
        self.numeralize_fn = functools.partial(generate_bert_pointwise_input,max_seq_len=self.max_seq_length,max_passage_len=self.max_passage_len,\
            tokenizer=self.tokenizer,device=self.device,wrap_tensor_flag=False)

        examples = [  (sample['question'],sample['passage']) for sample in self.sample_list]
        bert_input_t,seg_ids_t,input_mask_t = self.numeralize_fn(examples)
        for i,sample in enumerate(self.sample_list):
            sample.update({'input_ids':bert_input_t[i],'input_mask':input_mask_t[i],'segment_ids':seg_ids_t[i]})

    def add_bert_fields(self):
        bert_field = Field(batch_first=True, sequential=True, tokenize=lambda ids:[int(i) for i in ids],use_vocab=False, pad_token=0)
        self.fields+= [('input_ids',bert_field),('input_mask',bert_field),('segment_ids',bert_field)]

    def make_dataset(self):
        l = []
        for sample in  self.sample_list:
            l.append(Example.fromdict(sample,{t[0]:t for t in self.fields}))
        dataset = Dataset(l,self.fields)
        return dataset


    def make_batchiter(self,batch_size=32):
        dataset  = self.make_dataset()
        return Iterator(dataset,batch_size=batch_size,train=False,sort_key=lambda x: len(x.input_ids),sort_within_batch=True,device=self.device)
