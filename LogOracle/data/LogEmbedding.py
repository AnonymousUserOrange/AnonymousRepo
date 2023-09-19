from collections import Counter
import fasttext
import os
import numpy as np

def save_fasttext_vec(fasttext_model, vec_path):
    words = fasttext_model.get_words()
    with open(vec_path, 'w', encoding='utf-8') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(fasttext_model.get_dimension()) + "\n")
        # line by line, you append vectors to VEC file
        for w in words:
            v = fasttext_model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass

def get_token_embedding(fasttext_corpus, pure_template_token_emb_file, my_logger, config):
    my_logger.info('Training FastText model...')
    # pretrainedVectors = config.pretrained_embeddings_file
    my_logger.info("Start to train fasttext models with model %s, dim %d, lr %s, epoch %d, thread %d" %
                (config.fasttext_model, config.fasttext_dim, config.fasttext_lr, config.fasttext_epoch, config.fasttext_thread))
    text_model = fasttext.train_unsupervised(fasttext_corpus, minCount=1, maxn=10, wordNgrams=3,
                                             model=config.fasttext_model, dim=config.fasttext_dim,
                                             epoch=config.fasttext_epoch, lr=config.fasttext_lr, 
                                             thread=config.fasttext_thread)
    save_fasttext_vec(text_model, pure_template_token_emb_file)
    my_logger.info("Finish training fasttext vec. ")

def get_template_vec(module_pure_template_token_emb_file,
                     pure_templates_emb_file,
                     logs_emb_path, pure_template_file, my_logger):
    word_vocab = {}
    embedding_dim = -1
    my_logger.info("Read emb from %s" % module_pure_template_token_emb_file)
    with open(module_pure_template_token_emb_file, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            values = line.strip().split()
            if len(values) > 2:
                word, embed = values[0], np.asarray(values[1:], dtype=np.float)
                word_vocab[word] = embed
            else:
                embedding_dim = int(values[1])

    with open(pure_template_file, 'r', encoding="utf-8") as reader:
        template_tokens = [line.strip().split() for line in reader.readlines()]

    token_idf = {}
    idf_word_counter = Counter()
    total = len(template_tokens)
    for tokens in template_tokens:
        words = set(tokens)
        for word in words:
            idf_word_counter[word] += 1

    for word, count in idf_word_counter.most_common():
        token_idf[word] = np.log(total / count)

    with open(os.path.join(logs_emb_path, "token_idf.txt"), 'w', encoding='utf-8') as writer:
        for token, idf_score in token_idf.items():
            writer.write(' '.join([token, str(idf_score)]) + '\n')

    template_vec = []
    for tokens in template_tokens:
        place_holder = np.zeros(embedding_dim)
        if tokens[0] == "this_is_an_empty_event":
            template_vec.append(place_holder)
        else:
            word_counter = Counter(tokens)
            for token in tokens:
                if token in word_vocab.keys():
                    emb = word_vocab[token]
                else:
                    emb = np.zeros(embedding_dim)
                tf = word_counter[token] / len(tokens)
                if token in token_idf.keys():
                    idf_score = token_idf[token]
                else:
                    idf_score = 1
                place_holder += tf * idf_score * emb
            template_vec.append(place_holder)

    with open(pure_templates_emb_file, 'w', encoding='utf-8')as writer:
        writer.write(str(len(template_vec)) + ' ' + str(embedding_dim) + "\n")
        for embed in template_vec:
            embed = ' '.join([str(x) for x in embed.tolist()])
            writer.write(embed + "\n")
    my_logger.info("Save pure templates embeddings. ")

def read_vec(emb_file):
    vec = []
    with open(emb_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            values = line.strip().split()
            if len(values) == 2:
                key_count, dim = [int(x) for x in line.strip().split()]
            else:
                embedding = np.asarray(values[:], dtype=np.float)
                vec.append(embedding)
    assert len(vec) == key_count
    return vec

def save_testcase_vec(instance_emb_file, testcase_vec, my_logger):
    my_logger.info("Saving instance_vec. ")
    with open(instance_emb_file, 'w', encoding='utf-8') as writer:
        instance_id_list = sorted(list(testcase_vec.keys()))
        for instance_id in instance_id_list:
            embed = ' '.join([str(x) for x in testcase_vec[instance_id].tolist()])
            writer.write(' '.join([str(instance_id), embed]) + "\n")

def get_logs_embedding(logs_emb_path, instance_emb_file, pure_templates_emb_file, my_logger, id2template):
    testcase_vec = {}
    embedding_dim = 0
    template_vec = read_vec(pure_templates_emb_file)
    idf_log = {}
    idf_template_counter = Counter()
    total = len(id2template)
    for template_list in id2template.values():
        templates = set(template_list)
        for template_id in templates:
            idf_template_counter[template_id] += 1

    for template_id, count in idf_template_counter.most_common():
        idf_log[template_id] = np.log(total / count)

    with open(os.path.join(logs_emb_path, "idf_log.txt"), 'w', encoding='utf-8') as writer:
        for log_id, idf_score in idf_log.items():
            writer.write(' '.join([str(log_id), str(idf_score)]) + '\n')
    for instance_id, template_id_list in id2template.items():
        embedding_dim = len(template_vec[0])
        place_holder = np.zeros(embedding_dim)
        log_counter = Counter(template_id_list)
        for template_id in template_id_list:
            emb = template_vec[template_id]
            tf = log_counter[template_id] / len(template_id_list)
            if template_id in idf_log.keys():
                idf_score = idf_log[template_id]
            else:
                idf_score = 1
            place_holder += tf * idf_score * emb
        testcase_vec[instance_id] = place_holder

    save_testcase_vec(instance_emb_file, testcase_vec, my_logger)
    return testcase_vec