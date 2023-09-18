import torch
import javalang
from torch.autograd import Variable

class Instance():
    def __init__(self, log, trace, label):
        # self.tokens = [x.value for x in javalang.tokenizer.tokenize(log)]
        self.log = log.split('\n')
        self.trace = trace.split('\n')
        self.label = label

class TensorInstance():
    def __init__(self, input_seq_len, char_seq_len):
        self.input_context_char_seq = Variable(torch.LongTensor(input_seq_len, char_seq_len).zero_(),
                                               requires_grad=False)
        self.input_context_mask = Variable(torch.BoolTensor(input_seq_len).fill_(True), requires_grad=False)
        self.targets = Variable(torch.LongTensor(1), requires_grad=False)

    @property
    def inputs(self):
        return self.input_context_char_seq, self.input_context_mask

    @property
    def outputs(self):
        return self.targets

    def to_cuda(self, device):
        self.input_context_char_seq = self.input_context_char_seq.to(device)
        self.input_context_mask = self.input_context_mask.to(device)
        self.targets = self.targets.to(device)