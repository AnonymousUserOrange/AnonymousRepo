import os
import torch.nn as nn
import pickle
import numpy as np
import torch
import datetime
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report
from Codet5.Tranformers import EncodeTransformerBlock
from Codet5.dataloader import TextDataset


class MyEncoder(nn.Module):
    def __init__(self, args):
        super(MyEncoder, self).__init__()
        self.emb_dim = args.embedding_dim
        self.feed_forward_hidden = 2048
        self.encoder_layers = args.config.encoder_layers
        self.num_heads = 8
        self.input_dropout = args.config.input_dropout

        self.log_encoder = nn.ModuleList(
            [EncodeTransformerBlock(self.emb_dim, self.num_heads, self.feed_forward_hidden, self.input_dropout)
             for _ in range(self.encoder_layers)]
        )
        self.trace_embed = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.rnn = nn.GRU(input_size=args.embedding_dim, hidden_size=args.embedding_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.log_attn = nn.MultiheadAttention(embed_dim=args.embedding_dim,num_heads=1,batch_first=True)
        self.trace_attn = nn.MultiheadAttention(embed_dim=args.embedding_dim,num_heads=1,batch_first=True)
        self.classifier = RobertaClassificationHead(seq_len=512, hidden_size=args.embedding_dim, num_class=2)

    def forward(self, inputs):
        (logs_emb, logs_mask, trace_ids, trace_mask, labels, _) = inputs
        for layer in self.log_encoder:
            logs_emb = layer.forward(logs_emb, logs_mask)
        self.rnn.flatten_parameters()
        trace_seq = self.trace_embed(trace_ids.int())
        gru_hiddens, gru_states = self.rnn(trace_seq)
        trace_emb = trace_mask.unsqueeze(dim=-1) * gru_hiddens
        log_attn_embed, _ = self.log_attn(logs_emb,trace_emb,trace_emb,key_padding_mask=trace_mask)
        trace_attn_embed, _ = self.trace_attn(trace_emb,logs_emb,logs_emb,key_padding_mask=logs_mask)
        attn_embed = torch.cat((log_attn_embed,trace_attn_embed),dim=1)
        logits = self.classifier(attn_embed)
        probs = F.softmax(logits, dim=1)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, probs
        else:
            return probs


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, seq_len=512, hidden_size=512, num_class=2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(seq_len*2, num_class)

    def forward(self, hiddens):
        x = hiddens
        x = self.dropout(x)
        x = self.dense(x)
        x = x.squeeze(dim=-1)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=14, verbose=False, delta=0):
        """
        Args:
            save_path : model path to save
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        model_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(model_to_save, path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def model_train(train_dataset, eval_dataset, args, logger):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, #collate_fn=ds_collect_fn,
                                  batch_size=args.train_batch_size, num_workers=4)

    args.save_steps = 500
    args.max_steps = args.epochs * len(train_dataloader)
    args.warmup_steps = args.max_steps // 5

    model = MyEncoder(args)
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    best_acc = 0.0

    model.zero_grad()
    tb = SummaryWriter(log_dir='runs/codet5')

    output_dir = os.path.join('model', args.module)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    early_stopping = EarlyStopping(output_dir)
    eval_step = 0
    training_time = []
    for idx in range(args.epochs):
        training_start_time = datetime.datetime.now()
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        total_logits = []
        y_trues = []
        for step, batch in enumerate(bar):
            model.train()
            loss, logits = model(batch)
            total_logits.append(logits.detach().cpu().numpy())
            y_trues.append(batch[4].cpu().numpy())

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            tb.add_scalar("Loss", avg_loss, idx)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

        training_end_time = datetime.datetime.now()
        training_time.append((training_end_time - training_start_time).total_seconds())
        logger.info("***** Evaluate on Training dataset *****")

        # output result
        logits = np.concatenate(total_logits, 0)
        y_trues = np.concatenate(y_trues, 0)

        y_preds = []
        for logit in logits:
            y_preds.append(np.argmax(logit))

        logger.info("***** Training results *****")
        report = classification_report(y_trues, y_preds)
        print(report)
        tb.add_scalar("Train accuracy", classification_report(y_trues, y_preds, output_dict=True)['accuracy'], idx)

        results, eval_loss = evaluate(eval_dataset, model, args, logger)
        eval_step += 1
        tb.add_scalar("Eval accuracy", results["accuracy"], eval_step)

        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break
    avg_time_cost = sum(training_time) / len(training_time)
    logger.info(f'Training cost each epoch: {avg_time_cost}s')
    tb.close()


def evaluate(eval_dataset, model, args, logger):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, #collate_fn=ds_collect_fn,
                                 batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for step, batch in enumerate(bar):
        with torch.no_grad():
            lm_loss, logit = model(batch)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(batch[4].cpu().numpy())
        nb_eval_steps += 1
        avg_loss = round(eval_loss / nb_eval_steps, 5)
        bar.set_description("eval loss {}".format(avg_loss))
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))

    logger.info("***** Eval results *****")
    report = classification_report(y_trues, y_preds)
    print(report)

    return classification_report(y_trues, y_preds, output_dict=True), eval_loss


def model_test(test_dataset, model, args, logger):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, #collate_fn=ds_collect_fn,
                                 batch_size=args.eval_batch_size, num_workers=4)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    ids = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            lm_loss, logit = model(batch)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(batch[4].cpu().numpy())
            ids.append(batch[5].cpu())
        nb_eval_steps += 1

    # output result
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    ids = np.concatenate(ids, 0)
    
    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))

    report = classification_report(y_trues, y_preds)

    process_path = os.path.join('dataset', 'process', args.module, args.approach)
    check_file = os.path.join(process_path, 'FP_FN.txt')
    FP = []
    FN = []
    for y_id in range(len(y_preds)):
        if y_preds[y_id] == 1 and y_trues[y_id] == 0:
            FP.append(ids[y_id])
        elif y_preds[y_id] == 0 and y_trues[y_id] == 1:
            FN.append(ids[y_id])
    with open(check_file, 'w') as f:
        f.write(f'FP: ')
        for id in FP:
            f.write(str(id) + ' ')
        f.write('\n')
        f.write(f'FN: ')
        for id in FN:
            f.write(str(id) + ' ')
    print(report)


def train(args, logger):
    args.embedding_dim = 512
    process_path = os.path.join('dataset', 'process', args.module)
    process_neural_path = os.path.join(process_path, args.approach)
    if not os.path.exists(process_neural_path):
        os.makedirs(process_neural_path)
    tr_instance_file = os.path.join(process_neural_path, 'train_instance.pkl')
    val_instance_file = os.path.join(process_neural_path, 'valid_instance.pkl')
    vocab_file = os.path.join(process_neural_path, 'vocab.txt')

    with open(tr_instance_file, 'rb') as f:
        tr_item = pickle.load(f)
    with open(val_instance_file, 'rb') as f:
        val_item = pickle.load(f)
    with open(vocab_file, 'r') as f:
        vocab_size = int(f.read().strip())

    train_dataset = TextDataset(tr_item)
    eval_dataset = TextDataset(val_item)
    
    args.vocab_size = vocab_size
    model_train(train_dataset, eval_dataset, args, logger)
    

def test(args, logger):
    args.embedding_dim = 512
    model_path = os.path.join('model', args.module, 'best_network.pth')
    process_path = os.path.join('dataset', 'process', args.module)
    process_neural_path = os.path.join(process_path, args.approach)
    te_instance_file = os.path.join(process_neural_path, 'test_instance.pkl')
    vocab_file = os.path.join(process_neural_path, 'vocab.txt')
    with open(vocab_file, 'r') as f:
        vocab_size = int(f.read().strip())
    args.vocab_size = vocab_size
    model = MyEncoder(args)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    with open(te_instance_file, 'rb') as f:
        te_item = pickle.load(f)
    test_dataset = TextDataset(te_item)
    testing_start_time = datetime.datetime.now()
    model_test(test_dataset, model, args, logger)
    testing_end_time = datetime.datetime.now()
    test_time_cost = (testing_end_time - testing_start_time).total_seconds()
    avg_test_time = test_time_cost / len(test_dataset)
    logger.info(f'Testing time cost: {test_time_cost}s, average {avg_test_time}s')