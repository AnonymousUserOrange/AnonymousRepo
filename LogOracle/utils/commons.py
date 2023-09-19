from collections import Counter
import json
import os
import pickle
import random
import re
import shutil
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from data.GetLog import get_pure_templates, get_unduplicated_templates, load_templates, log_parsing
from data.LogEmbedding import get_template_vec, get_token_embedding, get_logs_embedding
from data.TraceEmbedding import get_trace_embedding, get_trace_event_embedding, get_tree_embedding
from data.vocab import CharVocab

    
class rawData(Dataset):
    def __init__(self, args, config, my_logger):
        self.args = args
        self.config = config
        # construct instance
        process_path = os.path.join('dataset','process',args.module)
        cache_path = os.path.join(process_path,'cache')
        instance_path = os.path.join(cache_path,'instance.json')
        rawLog_path = os.path.join('dataset/rawData',args.module,'log.json')
        module_log = os.path.join(process_path,'all_log.txt')
        trace_file = os.path.join(process_path,'trace.txt')
        test2id_file = os.path.join(process_path,'test2id.json')
        testcase_split_file = os.path.join(process_path,'testcase_split.txt')

        drain_path = os.path.join(process_path, "drain_result")
        drain_template_list_file = os.path.join(drain_path, "log_templates.txt")
        all_line_template_file = os.path.join(drain_path, "all_line_template.txt")
        pure_template_file = os.path.join(drain_path, "pure_templates.txt")
        clean_template_file = os.path.join(drain_path,"clean_templates.txt")
        config_path = os.path.join('config')
        config_file = os.path.join(config_path, "drain.ini")

        logs_emb_path = os.path.join(process_path,'log_embed')
        trace_emb_path = os.path.join(process_path,'trace_embed')

        instances = []
        
        # process log to templates
        if os.path.exists(instance_path):
            with open(instance_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    instances.append(json.loads(line))
            with open(test2id_file,'r', encoding="utf-8") as f:
                testcase2id = json.load(f)
                testcase_list = list(testcase2id.keys())
        else:
            rawLogs = []
            line2id = {}
            id2line = {}
            id2trace = {}
            id2testcase = {}
            id2rawLog = {}
            trace_list = []
            level_list = ['ALL','TRACE','DEBUG','INFO','WARN','ERROR','FATAL']
            with open(rawLog_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    rawLogs.append(json.loads(line))
            with open(module_log,'w',encoding='utf-8') as write_log:
                line_num = 0
                id = 0
                for rawLog in rawLogs:
                    testcase_level = []
                    id2line[id] = []
                    id2trace[id] = []
                    testcase = rawLog['testcase']
                    id2testcase[id] = testcase
                    id2rawLog[id] = rawLog['log']
                    for line in rawLog['log'].split('\n'):#TODO keep log same len with trace[:config.trace_seq_max_len]
                        line_tokens = line.strip().split()
                        if len(line_tokens) > 2 and re.match(r'\d{4}\-\d{2}-\d{2}', line_tokens[0]):
                            if line_tokens[2] not in level_list:
                                continue
                            log_content = ' '.join(line_tokens[4:])  # log content
                            trace = line_tokens[3]
                            testcase_level.append(line_tokens[2])  # level
                            write_log.write(log_content + "\n")
                            trace_list.append(trace)
                            line2id[line_num] = id
                            id2line[id].append(line_num)
                            id2trace[id].append(trace)
                            line_num+=1
                    id+=1
            trace_set = set(trace_list)
            with open(trace_file,'w',encoding='utf-8') as f:
                for trace in trace_set:
                    f.write(trace+'\n')
            if os.path.exists(drain_path):
                shutil.rmtree(drain_path)
            if not os.path.exists(drain_path):
                os.mkdir(drain_path)
            log_parsing(module_log, config_file, drain_template_list_file, all_line_template_file, my_logger)
            get_pure_templates(pure_template_file, drain_template_list_file, my_logger)
            get_unduplicated_templates(pure_template_file,all_line_template_file,clean_template_file,my_logger)
            all_line_template = load_templates(all_line_template_file)
            with open(clean_template_file,'r') as f:
                template_list = f.readlines()
            id2template = {}
            for id,line_list in id2line.items():
                template_id_list = [all_line_template[line] for line in line_list]
                id2template[id] = template_id_list

            # use unsupervised fasttext model and tf-idf to generate log embedding
            if os.path.exists(logs_emb_path):
                shutil.rmtree(logs_emb_path)
            if not os.path.exists(logs_emb_path):
                os.makedirs(logs_emb_path)
            pure_template_token_emb_file = os.path.join(logs_emb_path, "token_emb.vec")
            pure_templates_emb_file = os.path.join(logs_emb_path, "template_emb.vec")
            instance_emb_file = os.path.join(logs_emb_path,'instance_emb.vec')
            get_token_embedding(clean_template_file, pure_template_token_emb_file, my_logger, config)
            get_template_vec(pure_template_token_emb_file, pure_templates_emb_file,
                             logs_emb_path, clean_template_file, my_logger)
            # instance_log_vec = get_logs_embedding(logs_emb_path, instance_emb_file, pure_templates_emb_file, my_logger, id2template)

            # trace embedding 1.trace id 2.trace tree id
            trace2vec = {}
            trace2id = {}
            trace_id = 0
            for trace in trace_set:
                vec = np.zeros(config.trace_id_dim, dtype=np.uint8)
                vec[trace_id] = 1
                trace2id[trace] = trace_id
                trace2vec[trace] = vec
                trace_id+=1
                if trace_id==config.trace_id_dim:
                    break

            if os.path.exists(trace_emb_path):
                shutil.rmtree(trace_emb_path)
            if not os.path.exists(trace_emb_path):
                os.makedirs(trace_emb_path)
            trace_tree_file = os.path.join(trace_emb_path, "trace_tree.txt")
            trace_node_emb_file = os.path.join(trace_emb_path, "trace_node.vec")
            trace_event_emb_file = os.path.join(trace_emb_path,'trace_event.vec')
            trace_emb_file = os.path.join(trace_emb_path,'trace.vec')
            trace_node_vec = get_tree_embedding(trace_set,trace_tree_file,trace_node_emb_file,config)
            event_vec = get_trace_event_embedding(trace_set,trace_emb_path,trace_node_vec,trace_event_emb_file,config)
            instance_trace_vec = get_trace_embedding(event_vec,id2trace,trace_emb_file,config)

            id_now = 0
            instance = {'id':id_now,'log':'','rawLog':id2rawLog[id_now],'trace':'','trace_id':str(trace2id[trace_list[0]]),'trace_id_emb':'','testcase':id2testcase[id_now],
                        'trace_emb_tree':' '.join([str(x) for x in instance_trace_vec[id_now].tolist()]),
                        'log_id':' '.join([str(x) for x in id2template[id_now]]),
                        # 'log_emb':' '.join([str(x) for x in instance_log_vec[id_now].tolist()]),
                        'test':0 if rawLogs[id_now]['test']=='normal' else 1}
            trace_id_emb = np.zeros(config.trace_id_dim, dtype=np.uint8)
            for line_num,template_id in enumerate(all_line_template):
                clean_template = template_list[template_id]
                id = line2id[line_num]
                if id != id_now:
                    instance['trace_id_emb'] = ' '.join([str(x) for x in trace_id_emb.tolist()])
                    if instance['log']!= '':
                        instances.append(instance)
                    id_now = id
                    instance = {'id':id_now,'log':'','rawLog':id2rawLog[id_now],'trace':'','trace_id':str(trace2id[trace_list[line_num]]),'trace_id_emb':'','testcase':id2testcase[id_now],
                                'trace_emb_tree':' '.join([str(x) for x in instance_trace_vec[id_now].tolist()]),
                                'log_id':' '.join([str(x) for x in id2template[id_now]]),
                                # 'log_emb':' '.join([str(x) for x in instance_log_vec[id_now].tolist()]),
                                'test':0 if rawLogs[id_now]['test']=='normal' else 1}
                    trace_id_emb = trace2vec[trace_list[line_num]] if trace_list[line_num] in trace2vec.keys() else np.zeros(config.trace_id_dim)
                else:
                    instance['log']+=clean_template+'\n'
                    instance['trace']+=trace_list[line_num]+'\n'
                    if line_num!=0:
                        instance['trace_id']+=' '+str(trace2id[trace_list[line_num]])
                    trace_id_emb = trace_id_emb | trace2vec[trace_list[line_num]]
            instance['trace_id_emb'] = ' '.join([str(x) for x in trace_id_emb.tolist()])
            if instance['log']!= '':
                instances.append(instance)
            my_logger.info(f'instances number:{len(instances)}')
            # remove duplicate instance
            clean_instances = []
            clean_sim_instances = []
            for instance in instances:
                sim_instance = {'log':instance['log'],'trace':instance['trace'],'testcase':instance['trace'],'test':instance['test']}
                if sim_instance not in clean_sim_instances:
                    clean_sim_instances.append(sim_instance)
                    clean_instances.append(instance)
            noise_data = set()
            for i in range(len(clean_instances)):
                for j in range(i + 1, len(clean_instances)):
                    instance_a = {'log':clean_instances[i]['log'],'trace':clean_instances[i]['trace'],'test':clean_instances[i]['test']}
                    instance_b = {'log':clean_instances[j]['log'],'trace':clean_instances[j]['trace'],'test':clean_instances[j]['test']}
                    if(instance_a['test']==instance_b['test']):
                        continue
                    if all(instance_a[k] == instance_b[k] for k in instance_a.keys() if k != "test"):
                        noise_data.add(i)
                        noise_data.add(j)
            my_logger.warning(f'ratio of noise data/clean data: {len(noise_data)/len(clean_instances)}')
            for noise_idx in sorted(list(noise_data),reverse=True):
                clean_instances.pop(noise_idx)
            my_logger.info(f'clean instance number:{len(clean_instances)}')
            instances = clean_instances
            testcase2id = {}
            for inst_id, instance in enumerate(instances):
                if instance['testcase'] not in testcase2id.keys():
                    testcase2id[instance['testcase']] = [inst_id]
                else:
                    testcase2id[instance['testcase']].append(inst_id)
            testcase_list = list(testcase2id.keys())
            with open(instance_path, 'w', encoding='utf-8') as f:
                for instance in instances:
                    json.dump(instance,f)
                    f.write('\n')
            with open(test2id_file, 'w', encoding="utf-8") as f:
                json.dump(testcase2id,f)
        # split data to train/eval/test
        train_instance = []
        eval_instance = []
        test_instance = []
        train_instance_path = os.path.join(cache_path,'train_instances.json')
        eval_instance_path = os.path.join(cache_path,'eval_instances.json')
        test_instance_path = os.path.join(cache_path,'test_instances.json')
        if os.path.exists(train_instance_path):
            with open(train_instance_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    train_instance.append(json.loads(line))
            with open(eval_instance_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    eval_instance.append(json.loads(line))
            with open(test_instance_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    test_instance.append(json.loads(line))
        else:
            if os.path.exists(testcase_split_file):
                with open(testcase_split_file, 'r', encoding='utf-8') as f:
                    train_tc = f.readline().strip().split(':')[1].split(',')
                    eval_tc = f.readline().strip().split(':')[1].split(',')
                    test_tc = f.readline().strip().split(':')[1].split(',')
            else:
                test_tc = [testcase for testcase in testcase_list if 'ESTest' in testcase]
                testcase_list = [testcase for testcase in testcase_list if 'ESTest' not in testcase]
                train_portion = args.train_portion
                train_split = int(len(testcase_list) * train_portion)
                random.shuffle(testcase_list)
                train_tc, eval_tc = testcase_list[:train_split], testcase_list[train_split:]
                with open(testcase_split_file, 'w', encoding='utf-8') as f:
                    f.write('train:'+','.join(train_tc)+'\n')
                    f.write('eval:'+','.join(eval_tc)+'\n')
                    f.write('test:'+','.join(test_tc)+'\n')
            for testcase in train_tc:
                id_list = testcase2id[testcase]
                for id in id_list:
                    if len(instances[id]['log'].split('\n')) <= config.max_log_len:
                        train_instance.append(instances[id])
            for testcase in eval_tc:
                id_list = testcase2id[testcase]
                for id in id_list:
                    if len(instances[id]['log'].split('\n')) <= config.max_log_len:
                        eval_instance.append(instances[id])
            for testcase in test_tc:
                id_list = testcase2id[testcase]
                for id in id_list:
                    if len(instances[id]['log'].split('\n')) <= config.max_log_len:
                        test_instance.append(instances[id])
            with open(train_instance_path, 'w', encoding='utf-8') as f:
                for instance in train_instance:
                    json.dump(instance,f)
                    f.write('\n')
            with open(eval_instance_path, 'w', encoding='utf-8') as f:
                for instance in eval_instance:
                    json.dump(instance,f)
                    f.write('\n')
            with open(test_instance_path, 'w', encoding='utf-8') as f:
                for instance in test_instance:
                    json.dump(instance,f)
                    f.write('\n')
        self.train_instance = train_instance
        self.eval_instance = eval_instance
        self.test_instance = test_instance


class TextDataset(Dataset):
    def __init__(self, args, config, my_logger):
        self.instances = []
        self.target = []
        self.weights = []
        self.args = args
        self.config = config

        cache_path = os.path.join('dataset', 'process', args.module, 'cache')
        file_path = os.path.join(cache_path, args.phase + '_instances.json')
        weight_path = os.path.join(cache_path, 'cached_weight.pkl')

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        my_logger.info("Creating features from dataset file at %s", file_path)
        instances = []
        if not os.path.exists(file_path):
            process_dataset = rawData(args,config,my_logger)
            if args.phase == 'train':
                instances = process_dataset.train_instance
            elif args.phase == 'eval':
                instances = process_dataset.eval_instance
            elif args.phase == 'test':
                instances = process_dataset.test_instance
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    instances.append(json.loads(line))
        self.instances = instances
        label = [instance['test'] for instance in self.instances]
        self.target = label
        if args.phase == 'train':
            if os.path.exists(weight_path):
                with open(weight_path, 'rb') as f:
                    self.weights = pickle.load(f)
            else:
                class_sample_count = torch.tensor([Counter(label)[i] if Counter(label)[i] != 0 else torch.tensor([1.0e100]) for i in range(2)])
                class_weights = 1 / class_sample_count
                self.weights = torch.tensor(np.array([class_weights[instance['test']] for instance in self.instances]))
                with open(weight_path, 'wb') as f:
                    pickle.dump(self.weights, f)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item): 
        instance = self.instances[item]
        log = instance['log']
        log_len = len(log.split('\n'))
        trace_id_list = instance['trace_id'].strip().split()[:]
        trace_array = np.zeros((self.config.trace_seq_max_len,self.config.trace_id_dim),dtype=np.uint8)
        trace_mask = torch.BoolTensor(self.config.trace_seq_max_len).fill_(True)
        for trace_idx, trace_id in enumerate(trace_id_list[:self.config.trace_seq_max_len]):
            vec = np.zeros(self.config.trace_id_dim, dtype=np.uint8)
            trace_id = int(trace_id)
            vec[trace_id] = 1
            trace_array[trace_idx,:] = vec
            trace_mask[trace_idx] = False
        log_id_list = [int(log_id) for log_id in instance['log_id'].strip().split()[:self.config.max_log_len]]
        padding_log_list = np.pad(log_id_list,(0, self.config.max_log_len-len(log_id_list)), mode='constant', constant_values=-1)
        log_id = np.asarray(padding_log_list, dtype=np.int)
        trace_id_emb = np.asarray(instance['trace_id_emb'].strip().split()[:], dtype=np.float)
        target = instance['test']
        return (
            torch.tensor(target),
            torch.tensor(log_id),
            torch.tensor(trace_array,dtype=torch.float32),
            trace_mask,
            log_len
        )