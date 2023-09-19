
from collections import Counter
import os

import numpy as np


def get_tree_embedding(trace_set,trace_tree_file,trace_node_emb_file,config):
    trace_tree = []
    for trace in trace_set:
        trace_list = trace.strip().split('(')[0].split('.')
        for tree_depth, tree_node in enumerate(trace_list):
            if len(trace_tree)<tree_depth+1:
                trace_tree.append([])
            if tree_node not in trace_tree[tree_depth]:
                trace_tree[tree_depth].append(tree_node)
    with open(trace_tree_file,'w',encoding='utf-8') as f:
        for trace_depth in trace_tree:
            f.write(' '.join(trace_depth)+'\n')
    trace_node_vec = {}
    depth_len = 0
    for trace_depth, trace_list in enumerate(trace_tree):
        for node_id, trace_node in enumerate(trace_list):
            vec = np.zeros(config.tree_width)
            if node_id+depth_len<config.tree_width:
                vec[node_id+depth_len] = 1 
            trace_node_vec[str(trace_depth)+','+trace_node] = vec
        depth_len+=len(trace_list)
    with open(trace_node_emb_file,'w',encoding="utf-8") as f:
        for trace_node, vec in trace_node_vec.items():
            emb = ' '.join([str(x) for x in vec.tolist()])
            f.write(' '.join([trace_node,emb])+'\n')
    return trace_node_vec

def get_trace_event_embedding(trace_set,trace_emb_path,trace_node_vec,trace_event_emb_file,config):
    token_idf = {}
    idf_word_counter = Counter()
    total = len(trace_set)
    for trace in trace_set:
        trace_list = trace.strip().split('(')[0].split('.')
        for tree_depth, tree_node in enumerate(trace_list):
            idf_word_counter[str(tree_depth)+','+tree_node] += 1
    
    for word, count in idf_word_counter.most_common():
        token_idf[word] = np.log(total / count)
    
    with open(os.path.join(trace_emb_path, "token_idf.txt"), 'w', encoding='utf-8') as writer:
        for token, idf_score in token_idf.items():
            writer.write(' '.join([token, str(idf_score)]) + '\n')
    
    event_vec = {}
    for event in trace_set:
        place_holder = np.zeros(config.trace_dim)
        trace_list = event.strip().split('(')[0].split('.')
        depth_trace_list = [str(tree_depth)+','+tree_node for tree_depth, tree_node in enumerate(trace_list)]
        word_counter = Counter(depth_trace_list)
        for word in depth_trace_list:
            emb = trace_node_vec[word]
            tf = word_counter[word] / len(depth_trace_list)
            idf_score = token_idf[word]
            place_holder += tf * idf_score * emb
        event_vec[event.strip().split('(')[0]] = place_holder
    
    with open(trace_event_emb_file, 'w', encoding='utf-8')as f:
        for event, emb in event_vec.items():
            emb = ' '.join([str(x) for x in emb.tolist()])
            f.write(' '.join([event,emb])+'\n')
    return event_vec

def get_trace_embedding(event_vec,id2trace,trace_emb_file,config):
    trace_vec = {}
    idf_trace = {}
    idf_event_counter = Counter()
    total = len(id2trace)
    for trace in id2trace.values():
        event_set = set(trace)
        for event in event_set:
            idf_event_counter[event.strip().split('(')[0]] += 1
    
    for event, count in idf_event_counter.most_common():
        idf_trace[event] = np.log(total / count)
    
    for instance_id, trace in id2trace.items():
        place_holder = np.zeros(config.trace_dim)
        event_counter = Counter(trace)
        for event in trace:
            emb = event_vec[event.strip().split('(')[0]]
            tf = event_counter[event] / len(trace)
            idf_score = idf_event_counter[event.strip().split('(')[0]]
            place_holder += tf * idf_score * emb
        trace_vec[instance_id] = place_holder
    
    with open(trace_emb_file, 'w', encoding='utf-8') as f:
        instance_id_list = sorted(list(trace_vec.keys()))
        for instance_id in instance_id_list:
            embed = ' '.join([str(x) for x in trace_vec[instance_id].tolist()])
            f.write(' '.join([str(instance_id), embed]) + "\n")
            
    return trace_vec