## Model training

SAVE_DIR=./
SELF_TRAINING_ROOT=./self_training

conll-low-param:
	$(eval DATA_NAME=conll-2003)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=roberta)
	$(eval MODEL_NAME=roberta-base)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=900)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=300)
	$(eval HP_LABEL=5.9)
wikigold-low-param:
	$(eval DATA_NAME=wikigold)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=roberta)
	$(eval MODEL_NAME=roberta-base)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=300)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=500)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=200)
	$(eval HP_LABEL=5.9)
wnut-low-param:
	$(eval DATA_NAME=wnut-16)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=roberta)
	$(eval MODEL_NAME=roberta-base)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=900)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=450)
	$(eval HP_LABEL=5.9)
ncbi-low-param:
	$(eval DATA_NAME=ncbi-disease)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=bert)
	$(eval MODEL_NAME=dmis-lab/biobert-v1.1)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=900)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=300)
	$(eval HP_LABEL=5.9)
bc5cdr-low-param:
	$(eval DATA_NAME=bc5cdr)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=bert)
	$(eval MODEL_NAME=dmis-lab/biobert-v1.1)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=500)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=200)
	$(eval HP_LABEL=5.9)
chemdner-low-param:
	$(eval DATA_NAME=chemdner)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=bert)
	$(eval MODEL_NAME=dmis-lab/biobert-v1.1)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=900)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=300)
	$(eval HP_LABEL=5.9)

enzyme-fine-param:
	$(eval DATA_NAME=enzyme)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=roberta)
	$(eval MODEL_NAME=roberta-base)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=350)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=700)
	$(eval HP_LABEL=5.9)
astronomical-fine-param:
	$(eval DATA_NAME=astronomicalobject)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=roberta)
	$(eval MODEL_NAME=roberta-base)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=500)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=300)
	$(eval HP_LABEL=5.9)
award-fine-param:
	$(eval DATA_NAME=award)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=roberta)
	$(eval MODEL_NAME=roberta-base)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=350)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=400)
	$(eval HP_LABEL=5.9)
conference-fine-param:
	$(eval DATA_NAME=conference)
	$(eval TRAIN_ROOT=./data/annotated/$(DATA_NAME))
	$(eval MODEL_TYPE=roberta)
	$(eval MODEL_NAME=roberta-base)
	$(eval LR=1e-5)
	$(eval WEIGHT_DECAY=1e-4)
	$(eval EPOCH=50)
	$(eval SEED=0)
	$(eval ADAM_EPS=1e-8)
	$(eval ADAM_BETA1=0.9)
	$(eval ADAM_BETA2=0.98)
	$(eval WARMUP=200)
	$(eval TRAIN_BATCH=16)
	$(eval EVAL_BATCH=32)
	$(eval REINIT=0)
	$(eval BEGIN_STEP=200)
	$(eval LABEL_MODE=soft)
	$(eval PERIOD=100)
	$(eval HP_LABEL=5.9)

low-resource:
	$(eval EVAL_ROOT=./data/benchmarks/low-resource/$(DATA_NAME))

fine-grained:
	$(eval EVAL_ROOT=./data/benchmarks/fine-grained/$(DATA_NAME))

OUTPUT=$(SAVE_DIR)/outputs/$(DATA_NAME)/self_training/$(MODEL_TYPE)_reinit$(REINIT)_begin$(BEGIN_STEP)_period$(PERIOD)_$(LABEL_MODE)_hp$(HP_LABEL)_$(EPOCH)_$(LR)_SEED$(SEED)/

self-training:
	CUDA_DEVICE_ORDER=PCI_BUS_ID python3 $(SELF_TRAINING_ROOT)/run_self_training_ner.py \
    --train_dir $(TRAIN_ROOT) \
    --eval_dir $(EVAL_ROOT) \
    --model_type $(MODEL_TYPE) --model_name_or_path $(MODEL_NAME) \
    --learning_rate $(LR) \
    --weight_decay $(WEIGHT_DECAY) \
    --adam_epsilon $(ADAM_EPS) \
    --adam_beta1 $(ADAM_BETA1) \
    --adam_beta2 $(ADAM_BETA2) \
    --num_train_epochs $(EPOCH) \
    --warmup_steps $(WARMUP) \
    --per_gpu_train_batch_size $(TRAIN_BATCH) \
    --per_gpu_eval_batch_size $(EVAL_BATCH) \
    --logging_steps 100 \
    --save_steps 100000 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --output_dir $(OUTPUT) \
    --cache_dir $(SELF_TRAINING_ROOT)/pretrained_model \
    --seed $(SEED) \
    --max_seq_length 128 \
    --overwrite_output_dir \
    --self_training_reinit $(REINIT) --self_training_begin_step $(BEGIN_STEP) \
    --self_training_label_mode $(LABEL_MODE) --self_training_period $(PERIOD) \
    --self_training_hp_label $(HP_LABEL)

## Low-resource NER
conll-low: conll-low-param low-resource self-training
wikigold-low: wikigold-low-param low-resource self-training
wnut-low: wnut-low-param low-resource self-training
ncbi-low: ncbi-low-param low-resource self-training
bc5cdr-low: bc5cdr-low-param low-resource self-training
chemdner-low: chemdner-low-param low-resource self-training

## Fine-grained NER
enzyme-fine: enzyme-fine-param fine-grained self-training
astronomical-fine: astronomical-fine-param fine-grained self-training
award-fine: award-fine-param fine-grained self-training
conference-fine: conference-fine-param fine-grained self-training

