import numpy as np
from collections import Counter

from utils.data_utils import char2word


class InitialWeakData:
    def __init__(self):
        self.unlabeled_sentences = []
        self.name_list = []
        self.type_counter = {}
        self.skip_lowercase_ngram = []
        self.mined_phrases = []
        self.weak_labels = []
        self.abbreviations = []

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'type_counter':
                for d in value.keys():
                    if d not in self.type_counter.keys(): 
                        self.type_counter[d] = Counter()

                    for t in value[d].keys():
                        if t not in self.type_counter[d].keys():
                            self.type_counter[d][t] = value[d][t]
                        else:
                            self.type_counter[d][t] += value[d][t]
                continue

            elif key == 'skip_lowercase_ngram':
                if len(self.skip_lowercase_ngram) > 0:
                    for lstoken_index in range(len(value)):
                        value[lstoken_index][1] += self.skip_lowercase_ngram[-1][1]
            
            for attr in self.__dict__.keys(): 
                if attr == key:
                    self.__dict__[attr] += value

                    if key == 'name_list': 
                        self.name_list = list(set(self.name_list))


class EntityTypeSampler:
    def __init__(self, type_counter, stochastic_sampling):
        self.type_counter = type_counter
        self.stochastic_sampling = stochastic_sampling

    def __call__(self, mention):
        v = self.type_counter[mention]
        type_keys = list(v.keys())
        type_counts = [v[t_key] for t_key in type_keys]
        type_prior = [float(c) / float(sum(type_counts)) for c in type_counts]

        if self.stochastic_sampling:
            return np.random.choice(type_keys, 1, p=type_prior)[0]
        else:
            max_counts = max(type_counts)
            return np.random.choice([t_key for token_index, t_key in enumerate(type_keys) if type_counts[token_index] == max_counts])


def annotate_seqs(tokens, spans, phrases, entity_type):
    # B-I-O tagging scheme is used.
    ner_tags = ['O' for _ in range(len(tokens))]
    for p in phrases:
        start = int(p['start'])
        end = int(p['end'])
        name = p['name']
        assert end > start
        token_start = token_end = -1
        for i, (tk, sp) in enumerate(zip(tokens, spans)):
            if start >= sp[0] and start < sp[1]:
                token_start = i
            if end > sp[0] and end <= sp[1]:
                token_end = i

        if token_start != -1 and token_end != -1:
            for i in range(token_start, token_end+1):
                if i == token_start:
                    ner_tags[i] = 'B-' + entity_type
                else:
                    ner_tags[i] = 'I-' + entity_type
    return ner_tags


def split_phrases_by_ngram(phrases):
    phrases_for_each_ngram = {}
    phrases_startswith = {}

    max_ngram = 0

    for phrase in phrases: 
        splited = phrase.split()
        ngram = len(splited)

        if ngram not in phrases_for_each_ngram.keys():
            phrases_for_each_ngram[ngram] = []

        phrases_for_each_ngram[ngram].append(phrase)

        if ngram > max_ngram:
            max_ngram = ngram

        subtoken = splited[0]
        for lower_ngram in range(1, ngram):
            if lower_ngram > 1 :
                subtoken += ' ' + splited[lower_ngram-1]

            if lower_ngram not in phrases_startswith.keys():
                phrases_startswith[lower_ngram] = []

            phrases_startswith[lower_ngram].append(subtoken)

    for ngram in range(1, max_ngram+1):
        if ngram not in phrases_for_each_ngram.keys():
            phrases_for_each_ngram[ngram] = []

    return phrases_for_each_ngram, phrases_startswith, max_ngram


def detect_by_autophrase(mined_phrase, tokenized, lowercase_matching) :
    detected = []
    orig_sen = mined_phrase.replace('<phrase>', '').replace("</phrase>", '')
    splited = mined_phrase.split('<phrase>')

    offset = 0
    if len(splited) > 0:
        for s in splited:
            if '</phrase>' not in s:
                offset += len(s)
                continue

            token_p = s.split('</phrase>')[0].strip()

            if token_p != '':
                if lowercase_matching :
                    token_p = token_p.lower()
                detected.append([token_p, offset])

            offset += len(s.replace('</phrase>', ''))

    tokens, spans = char2word(orig_sen)

    for d_idx, phrase_info in enumerate(detected):
        phrase = phrase_info[0]
        phrase_span_start = phrase_info[1]

        start_flag = False
        for token_idx, span in enumerate(spans):
            start = span[0]
            end = span[1]

            if start == phrase_span_start:
                # change span idx => token idx
                detected[d_idx][1] = token_idx
                start_flag = True

            if end == phrase_span_start + len(phrase) and start_flag:
                detected[d_idx].append(token_idx + 1)
                break

    final_detected = []

    for phrase_info in detected:
        if len(phrase_info) != 3:
            # autophrase catch phrase shorter than our tokenized phrases. => ignore
            continue
        else:
            final_detected.append(phrase_info)

    return final_detected


def apply_autophrase(tokenized, matched_phrase, detected, token_index, token_index_to_tag, lowercase_matching):
    for phrase_info in detected:
        phrase = phrase_info[0]
        phrase_start = phrase_info[1]
        phrase_end = phrase_info[2]

        if phrase == matched_phrase and phrase_start == token_index:
            break

        if phrase_start <= token_index and phrase_end >= token_index + len(matched_phrase.split()):
            phrase_tokenized = ' '.join(tokenized[phrase_start:phrase_end])

            if lowercase_matching:
                phrase_tokenized = phrase_tokenized.lower()

            if phrase_tokenized != phrase and phrase_tokenized.replace(' ', '') != phrase.replace(' ', ''):
                # few annotation error in autophrase
                continue

            if phrase_start in token_index_to_tag.keys() and token_index_to_tag[phrase_start][0] == 'I':
                # concat to previous entity if overlapped
                token_index_to_tag[phrase_start] = 'I' + token_index_to_tag[token_index][1:]
            else :
                token_index_to_tag[phrase_start] = token_index_to_tag[token_index]

            for i in range(phrase_start + 1, phrase_end):
                token_index_to_tag[i] = 'I' + token_index_to_tag[phrase_start][1:]

    return token_index_to_tag


def apply_abbreviations(tokenized, token_index_to_tag, abbreviation):
    token_indicies_tagged = sorted(list(token_index_to_tag.keys()))
    # -100 will be used as the break signal
    token_indicies_tagged.append(-100)

    tagged_entity = ''
    tagged_type = ''

    abbrs_per_sen_keys = [' '.join(char2word(abbr_key)[0]) for abbr_key in abbreviation.keys()]
    abbrs_per_sen_values = [abbreviation[abbr_key] for abbr_key in abbreviation.keys()]

    for token_index in token_indicies_tagged:
        if tagged_entity == '':
            assert token_index_to_tag[token_index][0] == 'B'
            tagged_entity = tokenized[token_index]
            tagged_type = token_index_to_tag[token_index]

        elif token_index == -100 or token_index_to_tag[token_index][0] == 'B':
            if tagged_entity in abbrs_per_sen_keys:
                abbr_idx = abbrs_per_sen_keys.index(tagged_entity)
                abbr_splited = char2word(abbrs_per_sen_values[abbr_idx])[0]
                abbr_ntoken = len(abbr_splited)

                for a_idx in range(len(tokenized)-abbr_ntoken+1):
                    if tokenized[a_idx] == abbr_splited[0]:
                        # find and tag abbreviated expressions
                        if tokenized[a_idx:a_idx+abbr_ntoken] == abbr_splited:
                            if (a_idx in token_indicies_tagged and token_index_to_tag[a_idx][0] == 'I') or \
                                (a_idx+abbr_ntoken in token_indicies_tagged and token_index_to_tag[a_idx+abbr_ntoken][0] == 'I') :
                                # if tagged entity is ovelapped on abbreviated form, it would not be tagged.
                                continue

                            token_index_to_tag[a_idx] = 'B' + tagged_type[1:]
                            for a_idx_t in range(a_idx+1, a_idx+abbr_ntoken):
                                token_index_to_tag[a_idx_t] = 'I' + tagged_type[1:]

            if token_index != -100:
                tagged_entity = tokenized[token_index]
                tagged_type = token_index_to_tag[token_index]
        else :
            assert token_index_to_tag[token_index][0] == 'I'
            tagged_entity += ' ' + tokenized[token_index]
    
    return token_index_to_tag


