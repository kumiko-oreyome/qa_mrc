# beta : html text to paragraphs file ,sigle article , single question , quesion is append by author , just test file format
from common.util import jsonl_reader
def load_beta_file(path):
    print('load file %s'%(path))
    records = []
    for json_obj in jsonl_reader(path):
        question = json_obj['question']
        if len(question) == 0:
            continue
        for paragraph in json_obj['paragraphs']:
            records.append({'question':question,'passage':paragraph})
    return records


if __name__ == '__main__':
    from qa.reader import ReaderFactory
    from qa.judger import  MaxAllJudger
    from common.util import group_dict_list,RecordGrouper
    from qa.eval import QARankedListFormater
    reader_exp_name = 'reader/bert_default'
    sample_list = load_beta_file('./data/news2paragraph.jsonl')
    reader = ReaderFactory.from_exp_name(reader_exp_name)
    _preds = reader.evaluate_on_records(sample_list,batch_size=32)
    _preds = group_dict_list(_preds,'question')
    pred_answers  = MaxAllJudger().judge(_preds)
    pred_answer_list = RecordGrouper.from_group_dict('question',pred_answers).records
    ranked_list_formatter = QARankedListFormater(pred_answer_list)
    formated_result = ranked_list_formatter.format_result()
    with open('news_beta.txt','w',encoding='utf-8') as f:
        f.write('experiment settings\n')
        f.write('reader_exp_name : %s\n'%(reader_exp_name))
        f.write('##'*20)
        f.write('Content:\n\n')
        f.write(formated_result)