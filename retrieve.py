# This code is derived from the official DensePhrases repository and has been modified for NER dataset generation.
# https://github.com/princeton-nlp/DensePhrases

import json
import torch
import os
import random
import numpy as np
import logging
import spacy
import nltk
import inflect

from tqdm import tqdm
from scispacy.abbreviation import AbbreviationDetector

from densephrases.utils.single_utils import load_encoder
from densephrases.utils.open_utils import load_phrase_index, get_query2vec
from densephrases import Options

from utils.data_utils import load_gener_config
from utils.norm_utils import postproc_retrieved_data


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_all_query(questions, args, query_encoder, tokenizer, batch_size=64):
    query2vec = get_query2vec(
        query_encoder=query_encoder, tokenizer=tokenizer, args=args, batch_size=batch_size
    )

    all_outs = []
    for q_idx in tqdm(range(0, len(questions), batch_size)):
        outs = query2vec(questions[q_idx:q_idx+batch_size])
        all_outs += outs
    start = np.concatenate([out[0] for out in all_outs], 0)
    end = np.concatenate([out[1] for out in all_outs], 0)
    query_vec = np.concatenate([start, end], 1)
    logger.info(f'Query reps: {query_vec.shape}')
    return query_vec


def run_phrase_retrieval(args, questions, search_topks, mips=None, query_encoder=None, tokenizer=None, q_idx=None):
    # Load DensePhrases
    if query_encoder is None:
        logger.info(f'Query encoder will be loaded from {args.load_dir}')
        device = 'cuda' if args.cuda else 'cpu'
        query_encoder, tokenizer, _ = load_encoder(device, args)
    query_vec = embed_all_query(questions, args, query_encoder, tokenizer)

    # Load MIPS
    if mips is None:
        mips = load_phrase_index(args)

    # Only batch size 1 can be set in the current retrieval process. We will improve it later.
    step = 1

    logger.info(f'Aggergation strategy used: {args.agg_strat}')
    predictions = []
    evidences = []
    titles = []
    scores = []
    se_poss = []

    logger.info('Target questions ({}): {}'.format(len(questions), ', '.join(questions)))
    for q_idx in tqdm(range(0, len(questions), step)):
        search_topk = search_topks[q_idx]

        result = mips.search(
            query_vec[q_idx:q_idx+step],
            q_texts=questions[q_idx:q_idx+step], nprobe=args.nprobe,
            top_k=search_topk, max_answer_length=args.max_answer_length,
            aggregate=args.aggregate, agg_strat=args.agg_strat, return_sent=args.return_sent
        )
        prediction = [[ret['answer'] for ret in out][:search_topk] if len(out) > 0 else [''] for out in result]
        evidence = [[ret['context'] for ret in out][:search_topk] if len(out) > 0 else [''] for out in result]
        title = [[ret['title'] for ret in out][:search_topk] if len(out) > 0 else [['']] for out in result]
        score = [[ret['score'] for ret in out][:search_topk] if len(out) > 0 else [-1e10] for out in result]
        se_pos = [[(ret['start_pos'], ret['end_pos']) for ret in out][:search_topk] if len(out) > 0 else [(0,0)] for out in result]
        predictions += prediction
        evidences += evidence
        titles += title
        scores += score
        se_poss += se_pos
    
    return predictions, evidences, _, _, se_poss


def check_phrase(sent, phrase, start, end):
    if "PAR]" in phrase:
        return False
    if sent[start:end] != phrase:
        return False
    return True


def retrieve_phrases_and_sentences(args):
    if args.gener_config_path is None:
        raise ValueError("The argument 'gener_config_path' is missing.")

    gener_config = load_gener_config(args.gener_config_path)
    save_path = os.path.join(gener_config['retrieved_path'])
    os.makedirs(save_path)

    questions = []

    phrases_ignore = []
    plural_engine = inflect.engine()
    for subq_config in gener_config['subquestion_configs']:
        questions.append(subq_config['query_template'].replace('[TYPE]', subq_config['subtype']))

        if subq_config['remove_subtype']:
            phrases_ignore.append(subq_config['subtype'].lower())
            phrases_ignore.append(plural_engine.plural(subq_config['subtype'].lower()))
    
    search_topks = [max(int(1.2 * float(subq_config['top_k'])), subq_config['initial_top_k']) for subq_config in gener_config['subquestion_configs']]
    
    predictions, evidences, _, _, se_poss = run_phrase_retrieval(args, questions, search_topks)
 
    nlp = spacy.load('en_core_sci_sm')
    nlp.add_pipe('abbreviation_detector')
    
    total_sentences = total_phrases = 0
    for q_i, question in enumerate(questions):
        save_name = '_'.join(question.split())
        
        retrieved_data = []
        for i, se in tqdm(enumerate(se_poss[q_i])):
            offset = 0

            sent = evidences[q_i][i]
            phrase = predictions[q_i][i]
            start, end = se
            if not check_phrase(sent, phrase, start, end): continue
            
            json_example = {
                'sentence' : sent,
                'pos': [start, end],
                'phrase': phrase
            }
            
            retrieved_data.append(json_example)
        
        processed_data = postproc_retrieved_data(retrieved_data, nlp,
            gener_config['subquestion_configs'][q_i], phrases_ignore)
        
        top_k = gener_config['subquestion_configs'][q_i]['top_k']
        retrieved_sentences = retrieved_phrases = 0
        with open(os.path.join(save_path, '{}.json'.format(save_name)), 'w') as f:
            with open(os.path.join(save_path, '{}.raw'.format(save_name)), 'w') as f_raw :
                for s in processed_data.values():
                    f.write(json.dumps(s) + '\n')
                    f_raw.write(s['sentence'].strip() + "\n")
                    
                    retrieved_sentences += 1
                    retrieved_phrases += len(s['phrases'])

                    if retrieved_sentences >= top_k: break
        total_sentences += retrieved_sentences
        total_phrases += retrieved_phrases
    logger.info("{} sentences and {} phrases for {} subquestions have been retrieved.".format(total_sentences, total_phrases, len(questions)))


if __name__ == '__main__':
    # See options in densephrases.options
    options = Options()
    options.add_model_options()
    options.add_index_options()
    options.add_retrieval_options()
    options.add_data_options()
    # options for GeNER
    options.parser.add_argument('--gener_config_path', type=str, default=None, required=True, 
                    help="a pre-defined configuration file for data generation")
    args = options.parse()

    nltk.download('stopwords')

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    retrieve_phrases_and_sentences(args)


