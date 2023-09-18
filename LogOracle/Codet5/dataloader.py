import copy
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import re, string, pickle
import os, json
import torch
import random, math
import fasttext
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, item):
        self.item = item

    def __len__(self):
        return len(self.item)

    def __getitem__(self, offset):
        return self.item[offset]

def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message

    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    # s = " ".join([like_camel_to_tokens(word) for word in s.strip().split()])
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s

def not_empty(s):
    return s and s.strip()

def get_pure_templates(line):
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
                 
    if line != "":
        # Get pure templates.
        pure_line = re.sub(r'[^\w\d\/\_]+', ' ', line.strip())
        pure_line_token = pure_line.split()
        for i in range(len(pure_line_token)):
            if pure_line_token[i][0] == '/':
                pure_line_token[i] = ""
            else:
                # if bool(re.search(r'\d', pure_line_token[i])):
                #     pure_line_token[i] = ""
                if len(pure_line_token[i]) == 1:
                    pure_line_token[i] = ""
                else:
                    pure_line_token[i] = pure_line_token[i].lower()
        pure_line_token = list(filter(not_empty, pure_line_token))  # 除去 ''
        pure_line_token = list(filter(lambda x: x.lower() not in stopwords, pure_line_token))
        line = ' '.join(pure_line_token)
    return line

def balancing(x, y):
    minority_class = sum(y)  # 1
    majority_class = len(y) - minority_class  # 0
    bs = math.ceil(majority_class/5/minority_class)
    x_resampled = []
    y_resampled = []
    id_one = []
    for id in range(len(y)):
        if y[id]==0:
            x_resampled.append(x[id])
            y_resampled.append(y[id])
        else:
            id_one.append(id)
    sample_id = random.sample(id_one*bs, majority_class//5)
    for id in sample_id:
        x_resampled.append(x[id])
        y_resampled.append(y[id])
    return x_resampled, y_resampled

def data_slice(x, y):
    for inst_id in tqdm(range(len(x))):
        inst_x = x[inst_id]
        inst_y = y[inst_id]
        yield inst_x, inst_y

def load_supercomputers(args, process_path, log_file, process_neural_path):
    if args.embedding == 'codet5':
        from transformers import RobertaTokenizer, T5EncoderModel
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        encoder = T5EncoderModel.from_pretrained('Salesforce/codet5-small')
    elif args.embedding == 'bert':
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoder = BertModel.from_pretrained('bert-base-uncased')
    elif args.embedding == 'gpt2':
        from transformers import GPT2Tokenizer, GPT2Model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        encoder = GPT2Model.from_pretrained('gpt2')
    elif args.embedding == 'roberta':
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        encoder = RobertaModel.from_pretrained('roberta-base')
        
    print("Loading", log_file)
    vocab_file = os.path.join(process_neural_path, 'vocab.txt')
    raw_tr_instance_file = os.path.join(process_neural_path, 'raw_train_instance.json')
    raw_val_instance_file = os.path.join(process_neural_path, 'raw_valid_instance.json')
    raw_te_instance_file = os.path.join(process_neural_path, 'raw_test_instance.json')
    tr_instance_file = os.path.join(process_neural_path, 'train_instance.pkl')
    val_instance_file = os.path.join(process_neural_path, 'valid_instance.pkl')
    te_instance_file = os.path.join(process_neural_path, 'test_instance.pkl')

    train_instances = []
    valid_instances = []
    test_instances = []
    raw_train_instance = []
    raw_valid_instances = []
    raw_test_instances = []
    testcase2inst = {}
    test_tc = []
    rawLogs = []
    with open(log_file, "r", encoding='utf8') as f:
        for line in f:
            rawLogs.append(json.loads(line.strip()))
    instances = []
    inst_id = 0
    trace2id = {}
    trace_id = 1
    for rawLog_id, rawLog in enumerate(rawLogs):
        testcase = rawLog['testcase']
        label = rawLog['label']
        trace = rawLog['trace']
        trace_tensor = torch.zeros(args.config.max_log_len)
        for id, trace_method in enumerate(trace.split('\n')[:args.config.max_log_len]):
            if trace_method == '':
                continue
            if trace_method in trace2id.keys():
                trace_tensor[id] = trace2id[trace_method]
            else:
                trace2id[trace_method] = trace_id
                trace_tensor[id] = trace2id[trace_method]
                trace_id += 1
        instance = {'id': rawLog_id, 'testcase': testcase, 'label': label, 'log': rawLog['log'], 'trace': trace_tensor}
        content_len = len(rawLog['log'].split('\n'))
        if content_len > 0 and content_len <= args.config.max_log_len and 'ESTest' not in testcase:
            instances.append(instance)
            if testcase not in testcase2inst.keys():
                testcase2inst[testcase] = [inst_id]
            else:
                testcase2inst[testcase].append(inst_id)
            inst_id += 1
    testcase_split_file = os.path.join(process_path, 'testcase_split.txt')
    if os.path.exists(testcase_split_file):
        with open(testcase_split_file, 'r', encoding='utf-8') as f:
            train_tc = f.readline().strip().split(':')[1].split(',')
            eval_tc = f.readline().strip().split(':')[1].split(',')
            test_tc = f.readline().strip().split(':')[1].split(',')
    else:
        testcase_list = list(testcase2inst.keys())
        testcase_list = shuffle(testcase_list)
        # 直接按不同method划分
        train_split = int(0.8 * len(testcase_list))
        test_split = int(0.9 * len(testcase_list))
        if len(testcase_list) < 10:
            print("WARNING! Total number of testcase is less than 10!")
        train_tc, eval_tc, test_tc = testcase_list[:train_split], testcase_list[train_split:test_split], testcase_list[test_split:]
        with open(testcase_split_file, 'w', encoding='utf-8') as f:
            f.write('train:' + ','.join(train_tc) + '\n')
            f.write('eval:' + ','.join(eval_tc) + '\n')
            f.write('test:' + ','.join(test_tc) + '\n')
    train_log_set = set()
    duplicate_count = 0
    for testcase in train_tc:
        for inst_id in testcase2inst[testcase]:
            raw_inst = copy.deepcopy(instances[inst_id])
            raw_inst['trace'] = raw_inst['trace'].numpy().tolist()
            raw_train_instance.append(raw_inst)
            train_instances.append(instances[inst_id])
            train_log_set.add(hash(instances[inst_id]['log']))
    for testcase in eval_tc:
        for inst_id in testcase2inst[testcase]:
            raw_inst = copy.deepcopy(instances[inst_id])
            raw_inst['trace'] = raw_inst['trace'].numpy().tolist()
            raw_valid_instances.append(raw_inst)
            valid_instances.append(instances[inst_id])
    for testcase in test_tc:
        for inst_id in testcase2inst[testcase]:
            raw_inst = copy.deepcopy(instances[inst_id])
            raw_inst['trace'] = raw_inst['trace'].numpy().tolist()
            raw_test_instances.append(raw_inst)
            test_instances.append(instances[inst_id])
            if hash(instances[inst_id]['log']) in train_log_set:
                duplicate_count += 1
    print(f'duplicate test log appear in trainging set ratio: {duplicate_count / len(test_instances)}')
    for file, instance_list in ((raw_tr_instance_file, raw_train_instance), (raw_val_instance_file, raw_valid_instances),
                                (raw_te_instance_file, raw_test_instances)):
        with open(file, 'w', encoding='utf-8') as f:
            for instance in instance_list:
                json.dump(instance, f)
                f.write('\n')
    print(f'training instance number:{len(train_instances)}')
    print(f'valid instance number:{len(valid_instances)}')
    print(f'testing instance number:{len(test_instances)}')

    x_tr = [(instance['log'], instance['trace'], instance['id']) for instance in train_instances]
    y_tr = [instance['label'] for instance in train_instances]
    x_val = [(instance['log'], instance['trace'], instance['id']) for instance in valid_instances]
    y_val = [instance['label'] for instance in valid_instances]
    x_te = [(instance['log'], instance['trace'], instance['id']) for instance in test_instances]
    y_te = [instance['label'] for instance in test_instances]

    minority_class = sum(y_tr)  # 1
    majority_class = len(y_tr) - minority_class  # 0
    sampling_ratio = float(majority_class) / float(minority_class)
    if sampling_ratio > 5:
        x_tr, y_tr = balancing(x_tr, y_tr)

    num_train = len(x_tr)
    num_valid = len(x_val)
    num_test = len(x_te)
    num_total = num_train + num_valid + num_test
    num_train_pos = sum(y_tr)
    num_valid_pos = sum(y_val)
    num_test_pos = sum(y_te)
    num_pos = num_train_pos + num_valid_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Valid: {} instances, {} anomaly, {} normal' \
          .format(num_valid, num_valid_pos, num_valid - num_valid_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))
    
    tr_item = []
    val_item = []
    te_item = []
    encoder.to(args.device)
    E = {}
    trace_ids_list = []
    for x,y,item in ((x_tr, y_tr, tr_item), (x_val, y_val, val_item), (x_te, y_te, te_item)):
        for inst_x, inst_y in data_slice(x, y):
            logs = inst_x[0]
            trace_ids = inst_x[1]
            item_id = inst_x[2]
            trace_mask = torch.where(trace_ids==0, torch.tensor(0), torch.tensor(1)).to(torch.float)
            labels = inst_y
            logs_emb = torch.zeros(512, args.embedding_dim) # (log seq, emb dim)
            logs_mask = torch.zeros(512)  # log seq)
            for seq_id, log in enumerate(logs.split('\n')[:512]):
                if hash(log) in E.keys():
                    event_emb = E[hash(log)]
                else:
                    if args.embedding == 'gpt2':
                        inputs = tokenizer(log, return_tensors="pt", max_length=512, truncation=True)
                    else:
                        inputs = tokenizer(log, return_tensors="pt", max_length=512, padding='max_length', truncation=True)
                    if inputs['input_ids'].shape[-1] == 0:
                        continue
                    inputs.to(args.device)
                    outputs = encoder(**inputs)
                    event_emb = torch.mean(outputs.last_hidden_state.detach().cpu(), dim=1)[0] # token seq mean
                    E[hash(log)] = event_emb
                logs_emb[seq_id] = event_emb
                logs_mask[seq_id] = 1
            item.append((logs_emb, logs_mask, trace_ids, trace_mask, labels, item_id))
    
    with open(vocab_file, 'w') as f:
        f.write(str(trace_id))
    with open(tr_instance_file, 'wb') as f:
        pickle.dump(tr_item, f)
    with open(val_instance_file, 'wb') as f:
        pickle.dump(val_item, f)
    with open(te_instance_file, 'wb') as f:
        pickle.dump(te_item, f)
    return tr_item, val_item, te_item, trace_id
