import string
from nltk.corpus import stopwords
from collections import Counter

def strip_punct(s):
    while True:
        if s:
            if s[-1] in string.punctuation: s = s[:-1]
            elif s[0] in string.punctuation: s = s[1:]
            else: return s
        else: return ''


def add_space(s):
    if not s: return ''
    result = s[0]
    prev_c = s[0]
    for c in s[1:]:
        if prev_c in string.punctuation:
            result += ' '
            result += c
        else:
            if c in string.punctuation:
                result += ' '
                result += c
            else:
                result += c
        prev_c = c
    result = ' '.join(result.split())
    return result


def normalize_phrase(phrase, subq_config, ignore=[]):
    # Rule 1: split composite mentions based on 'and'
    if subq_config['split_composite_mention']:
        phrases = [p.strip() for p in phrase.split(' and ')]
    else:
        phrases = [phrase.strip()]

    # Rule 2: remove punctuations
    if subq_config['remove_punctuation']:
        phrases = [strip_punct(p).strip() for p in phrases if strip_punct(p).strip() != '']
    
    # Rule 3: remove lowercase phrases
    if subq_config['remove_lowercase_phrase']:
        phrases = [p for p in phrases if p != p.lower()]

    # Rule 4: remove the def 'the'
    if subq_config['remove_the']:
        phrases = [' '.join(p.split()[1:]) if p.split()[0].lower() == 'the' else p for p in phrases]

    # Rule 5: remove short phrase
    phrases = [p for p in phrases if len(p) >= subq_config['min_phrase_length']]
    
    # Rule 6: remove stopword
    if subq_config['remove_stopword']:
        phrases = [p for p in phrases if p.lower() not in stopwords.words('english')]
    
    # Rule 7: remove subtype
    if subq_config['remove_subtype']:
        phrases = [p for p in phrases if p.lower() not in ignore]
    
    # find the start positions of phrases
    if len(phrases) > 0:
        starts = [phrase.find(p) for p in phrases]
    else:
        starts = None

    return phrases, starts


def postproc_retrieved_data(retrieved_data, nlp, subq_config, ignore=[]):
    # This code normalizes retrieved phrases, detects abbreviations, and merges duplicate sentences
    
    processed_data = {}

    for json_example in retrieved_data:
        sent = json_example['sentence']
        pos = json_example['pos']
        phrase = json_example['phrase']

        if sent not in processed_data.keys():
            processed_data[sent] = {
                'sentence': sent,
                'poss': [pos],
                'phrases': [phrase]
            }
        else :
            processed_data[sent]['poss'].append(pos)
            processed_data[sent]['phrases'].append(phrase)

    for sent in processed_data.keys():
        json_example = processed_data[sent]
        
        abbreviations = {}
        doc = nlp(json_example['sentence'])
        for abrv in doc._.abbreviations:
            if abrv.text.lower() not in ignore :
                abbreviations[abrv._.long_form.text] = [abrv.text, (abrv.start_char, abrv.end_char)]

        processed_data[sent]['abbreviations'] = abbreviations

        phrases = []
        pos_mark = [0 for _ in range(len(json_example['sentence']))]
        for phrase, pos in zip(json_example['phrases'], json_example['poss']):
            normalized_names, starts = normalize_phrase(phrase, subq_config, ignore)

            if normalized_names == []:
                pass
            else:
                # If there are conflicts between spans within duplicate sentences, the higher-ranked span is selected.
                for nn, st in zip(normalized_names, starts):
                    start_idx = pos[0] + st

                    if nn != json_example['sentence'][start_idx:start_idx+len(nn)]:
                        continue

                    if sum(pos_mark[start_idx:start_idx+len(nn)]) == 0:
                        phrases.append({
                            'name': nn,
                            'start': start_idx,
                            'end': start_idx+len(nn)
                        })

                        for i in range(start_idx, start_idx+len(nn)):
                            pos_mark[i] = 1
                        
        processed_data[sent]['phrases'] = phrases
        del processed_data[sent]['poss']
    
    return processed_data


def preproc_dictionary(dictionary, min_phrase_frequency=0, lowercase_matching=True):
    # This code pre-processes a raw pseudo-dictionary for dictionary matching.
    type_counter = {}

    phrases = [p.split("\t")[0].strip() for p in dictionary]

    # add spaces before and after punctuations
    phrases = [add_space(p) for p in phrases]

    if lowercase_matching:
        phrases = [p.lower() for p in phrases]

    # type of dictionary
    t = dictionary[0].split('\t')[1].strip()
    for p in phrases:
        if p not in type_counter.keys(): type_counter[p] = Counter()
        type_counter[p][t] += 1

    # Remove duplicates
    phrases = list(set(phrases))

    processed_phrases = []

    cnt = 0
    for p in phrases:
        if max(type_counter[p].values()) > min_phrase_frequency:
            processed_phrases.append(p)
            cnt += 1

    return processed_phrases, type_counter


