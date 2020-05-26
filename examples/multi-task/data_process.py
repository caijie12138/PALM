import json
import os
import csv
import re
import pandas as pd

def prepare_data(data_dir, word_dict = {'<unk>':0}):
    train_path = os.path.join(data_dir, 'data_for_zhouding_train.txt')
    dev_path = os.path.join(data_dir, 'data_for_zhouding_dev.txt')
    test_path = os.path.join(data_dir, 'data_for_zhouding_test.txt')

    train_data = data_process(train_path, word_dict)
    dev_data = data_process(dev_path, train_data[-3], types=train_data[-2], topics=train_data[-1])
    test_data = data_process(test_path, dev_data[-3], types=train_data[-2], topics=train_data[-1])

    return train_data, dev_data, test_data, test_data[-1]

def write_data(all_data,path):
    data_tag = ''
    type_label = 0
    
    data_type = path.split('/')[-2]
    if data_type == 'type':
        type_label = 1
    elif data_type == 'topic':
        type_label = 2

    for index,data in enumerate(all_data):
        if index == 0: 
           data_tag = 'train'
        elif index == 1: 
           data_tag = 'dev'
        else:
           data_tag = 'test'

        if not os.path.exists(path):
            os.mkdir(path)
 
        csv_path = os.path.join(path, data_tag + '.csv')
        tsv_path = os.path.join(path, data_tag + '.tsv')
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["label", "text_a"])
            for context,label in data[type_label].items():
                csv_writer.writerow([label,str(context)])
            
            # csv to tsv
            csv_read = pd.read_csv(csv_path)
            with open(tsv_path, 'w') as write_tsv:
                write_tsv.write(csv_read.to_csv(sep='\t', index=False))


def data_process(path,word_dict,types=None,topics=None):
    data_for_goal_transfer = {}
    data_for_goal_type = {}
    data_for_goal_topic = {}
    if types == None and topics == None:
        types = {}
        topics = {}
    with open(path,'r') as f:
        # context = ''
        for line in f.readlines():
            if line == '\n':
                continue
            units = line.strip().split('\t')
            utterence = units[0]

            if units[2] not in types:
                types[units[2]] = len(types)
            if units[3] not in topics:
                topics[units[3]] = len(topics)

            for word in utterence.split():
                if word not in word_dict:
                    word_dict[word] = len(word_dict)

            if units[0].startswith('['):
                utterence = units[0][4:]

            # context += ' '+utterence

            if units[-1] == 'Bot':
                data_for_goal_transfer[utterence] = units[1]
                data_for_goal_type[utterence] = types[units[2]]
                data_for_goal_topic[utterence] = topics[units[3]]

    return [data_for_goal_transfer,data_for_goal_type,data_for_goal_topic,word_dict,types,topics]



if __name__=='__main__':

    data_path = './data/dialog/'
    
    #test_data[-1] for word dict, it's unnecessary if pre-train model is used 
    #train_data,dev_data,test_data,test_data[-1] = prepare_data(data_path)
    train_data, dev_data, test_data, _ = prepare_data(data_path)

    write_data([train_data,dev_data,test_data],'./data/dialog/type/') 
    write_data([train_data,dev_data,test_data],'./data/dialog/topic/')
