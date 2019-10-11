#def load_bert_model(archive_path,cls,device=None,weight_path=None):
    #output_model_file = "../model_dir/best_model"
    #output_config_file = "../model_dir/chinese_wwm_ext_pytorch/bert_config.json"   
#    config = BertConfig(archive_path)
#    model = BertForQuestionAnswering(config)
#    model.load_state_dict(torch.load(output_model_file,map_location=args.device))
#    if device is not None:
#        return model.to(device)
#    return model.cpu()


#   tokenizer =  BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
#    cvt =  BertInputConverter(tokenizer)
