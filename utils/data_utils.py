import string
import json
import os
import numpy as np


def convert_to_bond_format(jsonl_data, tag2id=None):
    if tag2id == None :
        tag2id = {"O" : 0}
        ner_tags = sorted(list(set([tag for d in jsonl_data for tag in d['ner_tags'] if tag != "O"])))
        for t_idx, tag in enumerate(ner_tags) :
            tag2id[tag] = t_idx+1

    result = []

    for d in jsonl_data :
        result.append({
            "str_words" : d['tokens'],
            "tags" : [tag2id[tag] for tag in d['ner_tags']]
        })

    return result, tag2id


def get_json_line(tokenized, token_index_to_tag, line_idx) : 
    json_format = {"id" : line_idx, "document_id" : line_idx, "ner_tags" : [], "tokens" : [], "spans" : []}

    for idx in range(len(tokenized)) :
        obj = tokenized[idx]
        json_format['tokens'].append(obj)
        if idx not in token_index_to_tag.keys() :
            json_format['ner_tags'].append("O")
        else :
            json_format['ner_tags'].append(token_index_to_tag[idx])
        if len(json_format['spans']) == 0 :
            json_format['spans'].append([0, len(obj)])
        else :
            start = json_format['spans'][-1][1]+1
            json_format['spans'].append([start, start+len(obj)])

    return json_format


def load_gener_config(config_path) :
    config = json.load(open(config_path))

    default_config_path = config_path.replace(os.path.basename(config_path), 'default.json')
    
    if os.path.exists(default_config_path) :
        default_config = json.load(open(default_config_path))

        for q_idx, q_dic in enumerate(config['subquestion_configs']) :
            for d_key, d_info in default_config['default'].items() :
                if d_key not in q_dic.keys() :
                    config['subquestion_configs'][q_idx][d_key] = d_info
    else:
        raise FileNotFoundError("Cannot find the 'default.json' file. Add this file to your 'configs' directory.")
    return config


def char2word(seq):
    spans = []
    tokens = []

    word_tmp = []
    start, end, offset = 0,0,0
    prev_c = "NOTHING"
    for c_i, c in enumerate(seq):
        if c.isspace():  # whitespace
            if word_tmp:
                # append the token
                end = start+len(word_tmp)
                tokens.append("".join(word_tmp))
                spans.append([start, end])
                word_tmp = []
        else:
            if not word_tmp: 
                start = offset
            if c in string.punctuation:
                # append the token
                if word_tmp:
                    end = start+len(word_tmp)
                    tokens.append("".join(word_tmp))
                    spans.append([start, end])
                    
                    start = offset

                # append the current punctuation
                tokens.append(c)
                end = start + 1
                spans.append([start, end])
                
                word_tmp = []
            else:
                if prev_c in string.punctuation:
                    # append the previous punctuation
                    #start = offset
                    word_tmp.append(c)
                else:
                    word_tmp.append(c)

        prev_c = c
        offset += 1

    # Last token left
    if word_tmp:
        # append the token
        end = start+len(word_tmp)
        tokens.append("".join(word_tmp))
        spans.append([start, end])

    # check
    assert len(tokens) == len(spans)
    for tk, sp in zip(tokens, spans):
        assert seq[sp[0]:sp[1]] == tk
    return tokens, spans


