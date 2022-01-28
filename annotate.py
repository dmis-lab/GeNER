import argparse
import json
import os
import numpy as np

from collections import Counter
from tqdm import tqdm

from utils.data_utils import load_gener_config, char2word, get_json_line, convert_to_bond_format
from utils.norm_utils import add_space, preproc_dictionary
from utils.labeling_utils import InitialWeakData, EntityTypeSampler, annotate_seqs, split_phrases_by_ngram, apply_autophrase, apply_abbreviations, detect_by_autophrase


def dictionary_matching(unlabeled_sentences, name_list, type_counter, entity2type, lowercase_matching, skip_lowercase_ngram, refine_boundary, mined_phrases, weak_labels, abbreviations):
    generated_data = []

    # We search phrases from 1-gram to n-gram for compuational cost prob
    phrases_for_each_ngram, phrases_startswith, max_ngram = split_phrases_by_ngram(name_list)

    lstoken_index = 0
    for line_idx, tokenized in tqdm(enumerate(unlabeled_sentences)):
        token_index_to_tag = {}
        
        if refine_boundary:
            detected = detect_by_autophrase(mined_phrases[line_idx], tokenized, lowercase_matching)

        if weak_labels != None:
            assert len(weak_labels[line_idx]) == len(tokenized)

            for idx in range(len(weak_labels[line_idx])):
                if weak_labels[line_idx][idx] != 'O':
                    token_index_to_tag[idx] = weak_labels[line_idx][idx]

        if skip_lowercase_ngram != None:
            if line_idx >= skip_lowercase_ngram[lstoken_index][1]:
                lstoken_index += 1

        for token_index in range(len(tokenized)):
            matched_phrase = ''

            if token_index in token_index_to_tag.keys():
                continue

            candidate = tokenized[token_index]
            for ngram in range(1, max_ngram+1):

                if token_index+ngram > len(tokenized) or token_index+ngram-1 in token_index_to_tag.keys():
                    break

                if ngram > 1:
                    # Search candidate by increasing the ngram of phrases for given position.
                    candidate += ' ' + tokenized[token_index+ngram-1].strip()
                
                # Rule 9
                if ngram <= skip_lowercase_ngram[lstoken_index][0]:
                    if candidate == candidate.lower():
                        continue

                if lowercase_matching:
                    candidate = candidate.lower()

                if candidate in phrases_for_each_ngram[ngram]:
                    tag_postfix = '-' + entity2type(candidate)

                    for i in range(token_index, token_index+ngram):
                        if i == token_index:
                            token_index_to_tag[i] = 'B' + tag_postfix
                        else:
                            token_index_to_tag[i] = 'I' + tag_postfix
                    
                    matched_phrase = candidate

                if ngram < max_ngram and candidate not in phrases_startswith[ngram]:
                    # stop searching since there is no phrases starts with candidate
                    break

            if matched_phrase != '':
                if refine_boundary:
                    token_index_to_tag = apply_autophrase(tokenized, matched_phrase, detected, token_index, token_index_to_tag, lowercase_matching)

        if len(abbreviations[line_idx]) > 0 and len(token_index_to_tag.keys()) > 0:
            token_index_to_tag = apply_abbreviations(tokenized, token_index_to_tag, abbreviations[line_idx])

        generated_data.append(get_json_line(tokenized, token_index_to_tag, line_idx))

    return generated_data


def generate_initial_data(load_path, target_subq):
    dictionary = []
    unlabeled_sentences = []
    weak_labels = []
    mined_phrases = []
    abbreviations = []

    load_file = '{}.json'.format(target_subq['subquestion'])

    examples = []
    with open(os.path.join(load_path, load_file)) as f:
        json_examples = f.readlines()
        
        print(load_file)
        for i, j_ex in enumerate(json_examples):
            json_example = json.loads(j_ex)
            examples.append(json_example)

    # AutoPhrase
    if target_subq['refine_boundary']:
        autophrase_file = '{}.autophrase'.format(target_subq['subquestion'])
        if os.path.isfile(os.path.join(load_path, autophrase_file)):
            
            with open(os.path.join(load_path, autophrase_file)) as f:
                mined_phrases = [l.strip() for l in f.readlines()]
        else:
            raise FileNotFoundError("Cannot find the '{} file.".format(autophrase_file))
    else:
        mined_phrases = ['' for _ in range(len(examples))]

    for ex in examples:
        tokens, spans = char2word(ex['sentence'])
        phrases = ex['phrases']

        for phrase in phrases:
            dictionary.append('{}\t{}'.format(phrase['name'], target_subq['type']))

        ner_tags = annotate_seqs(tokens, spans, phrases, target_subq['type'])
        
        unlabeled_sentences.append(tokens)
        weak_labels.append(ner_tags)

        if target_subq['add_abbreviation']:
            abbrs = {}
            for ent in ex['abbreviations'].keys():
                abb = ex['abbreviations'][ent][0]
                abbrs[ent] = abb

            abbreviations.append(abbrs)
        else:
            abbreviations.append({})

    assert len(weak_labels) == len(unlabeled_sentences)
    assert len(mined_phrases) == len(unlabeled_sentences)
    assert len(abbreviations) == len(unlabeled_sentences)
    return unlabeled_sentences, weak_labels, mined_phrases, abbreviations, dictionary


def main(args, gener_config):
    add_abbreviation = gener_config['add_abbreviation']
    refine_boundary = gener_config['refine_boundary']

    initial_data = InitialWeakData()

    load_path = gener_config['retrieved_path']
    
    # obtain an initial dataset (i.e., dataset before the dictionary matching process)
    for subq_config in gener_config['subquestion_configs']:
        target_subq = subq_config
        target_subq['subquestion'] = subq_config['query_template'].replace('[TYPE]', subq_config['subtype']).replace(' ', '_')
        target_subq['add_abbreviation'] = add_abbreviation
        target_subq['refine_boundary'] = refine_boundary
        
        unlabeled_sentences, weak_labels, mined_phrases, abbreviations, raw_dictionary = generate_initial_data(load_path, target_subq)

        name_list, type_counter = preproc_dictionary(raw_dictionary, subq_config['min_phrase_frequency'], args.lowercase_matching)
        
        skip_lowercase_ngram = [[target_subq['skip_lowercase_ngram'], len(unlabeled_sentences)]]

        initial_data.update(name_list=name_list, unlabeled_sentences=unlabeled_sentences, weak_labels=weak_labels, mined_phrases=mined_phrases, \
            abbreviations=abbreviations, type_counter=type_counter, skip_lowercase_ngram=skip_lowercase_ngram)

    entity2type = EntityTypeSampler(initial_data.type_counter, args.stochastic_sampling)
    
    # conduct dictionary matching and obtain the final NER dataset
    generated_data = dictionary_matching(**(initial_data.__dict__), entity2type=entity2type, lowercase_matching=args.lowercase_matching, refine_boundary=refine_boundary)
    
    # (Hugging Face) Transformers data format
    with open(os.path.join(gener_config['annotated_path'], 'train_hf.json'), 'w') as g:
        for ex in generated_data:
            g.write(json.dumps(ex) + '\n')
     
    # BOND [Liang et al., 2020] data format
    bond_data, _ = convert_to_bond_format(generated_data)
    json.dump(bond_data, open(os.path.join(gener_config['annotated_path'], 'train.json'), 'w'))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowercase_matching', type=bool, default=True)
    parser.add_argument('--stochastic_sampling', type=bool, default=True)
    parser.add_argument('--gener_config_path', type=str, required=True)

    args = parser.parse_args()

    np.random.seed(0)

    gener_config = load_gener_config(args.gener_config_path)
    
    # check if the same directory exists
    os.makedirs(gener_config['annotated_path'])
    
    main(args, gener_config)


