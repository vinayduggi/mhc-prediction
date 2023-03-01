import logging
import torch
import re
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import json


logging.basicConfig(filename='train_logs',
                    filemode='w',
                    format='%(asctime)s :: %(levelname)s :: %(funcName)s :: %(name)s :: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def load_model():
    """load a model onto GPU/CPU """
    PRE_TRAINED_MODEL = 'Rostlab/prot_bert'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Device Found {:<10}'.format(str(device)))
    model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels = 2, output_attentions = False, output_hidden_states = False)
    model.classifier = torch.nn.Linear(1024, 1)
    model.to(device)
    params = list(model.named_parameters())
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL, do_lower_case=False)
    logger.info('Model and Tokenizer Load Successful {:<30}'.format(PRE_TRAINED_MODEL))
    logger.info('The ProtBert model has {:} different named parameters.\n'.format(len(params)))
    logger.info('==== Embedding Layer ====\n')
    for p in params[0:5]:
        logger.info('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))

    logger.info('\n==== First Transformer ====\n')

    for p in params[5:21]:
        logger.info('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))
        
    logger.info('\n==== Output Layer ====\n')

    for p in params[-4:]:
        logger.info('{:<55} {:>12}'.format(p[0], str(tuple(p[1].size()))))
    
    return model, tokenizer, device


class SequenceTokenizer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(self.file_name, low_memory=False)
        
        hlallelemapping = open('hla_allele_classI.json',)
        hlaDict = json.load(hlallelemapping)
        for key in hlaDict.keys():
            self.data.loc[self.data['mhc_allele']==key, 'hla_sequence'] = hlaDict[key]
        
        relevant_cols = ['mhc_allele', 'peptide', 'bind_class','label', 'hla_sequence']
        self.data = self.data[relevant_cols]



    def PreProcess_Tokenize_Encode(self, tokenizer):
        peptide_hla_cleaned_list = []
        label_list = self.data['label'].tolist()
        peptide_list = self.data['peptide'].tolist()
        hla_sequence_list = self.data['hla_sequence'].tolist()
        peptides_spaced_list = [" ".join(seq) for seq in peptide_list]
        peptides_cleaned_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in peptides_spaced_list]       
        hla_sequence_spaced_list = [" ".join(seq) for seq in hla_sequence_list]
        hla_sequence_cleaned_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in hla_sequence_spaced_list]
        logger.info('Adding Special Tokens CLS and SEP for human_mhc_classI peptides and sequences...')
        
        for peptide, hla_seq in zip(peptides_cleaned_list, hla_sequence_cleaned_list):
            peptide_st = "[CLS] {0} [SEP]".format(peptide)
            hla_st = "{0} [SEP]".format(hla_seq)
            peptide_hla = " ".join([peptide_st, hla_st])
            peptide_hla_cleaned_list.append(peptide_hla)
            
        logger.info('DONE.Adding Special Tokens CLS and SEP for human_mhc_classI peptides and sequences...')
        logger.info('Tokenizing {:,} human_mhc_classI peptides and sequences...'.format(len(peptide_hla_cleaned_list)))
        encode_peptides = tokenizer.batch_encode_plus(
        peptide_hla_cleaned_list,
        add_special_tokens = False,
        max_length = 560, 
        padding = 'max_length', 
        return_attention_mask = True, 
        return_tensors='pt'
        )
        input_ids = encode_peptides['input_ids']
        attention_mask = encode_peptides['attention_mask']
        targets = torch.tensor(label_list, dtype=torch.float)
        targets = targets.unsqueeze(1)
        logger.info('DONE. Tokenizing {:,} human_mhc_classI peptides and sequences...'.format(len(input_ids)))

        return input_ids, attention_mask, targets
    
            
class TrainValidSplit:
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


    def TrainValidationSplit(self):
        dataset = TensorDataset(self.input_ids, self.attention_mask, self.labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        logger.info('{:>5,} training samples'.format(train_size))
        logger.info('{:>5,} validation samples'.format(val_size))

        return train_dataset, val_dataset


    def BatchIterator(self):
        train_dataset, val_dataset = self.TrainValidationSplit()
        batch_size = 64

        train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset),
                batch_size = batch_size
            )

        validation_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset),
                batch_size = batch_size
            )

        return train_dataloader, validation_dataloader


def OptimizeLearningRateSchedule(model, input_ids, attention_mask, labels, train_dataloader):
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

    epochs = 4
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    return scheduler, optimizer, epochs


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    
    elapsed_rounded = int(round((elapsed)))
    
    
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(file_name):
    model,tokenizer, device = load_model()
    preprocess = SequenceTokenizer(file_name)
    input_ids, attention_mask, labels = preprocess.PreProcess_Tokenize_Encode(tokenizer)
    T = TrainValidSplit(input_ids, attention_mask, labels)
    train_dataloader,validation_dataloader = T.BatchIterator()
    scheduler, optimizer, epochs = OptimizeLearningRateSchedule(model, input_ids, attention_mask, labels, train_dataloader)
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        logger.info('')
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('Training...')
        t0 = time.time()

        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

        
            if step % 40 == 0 and not step == 0:
            
                elapsed = format_time(time.time() - t0)
            
            
                logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            model.zero_grad()        


            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        return_dict=True)

            loss = result.loss
            logits = result.logits


            total_train_loss += loss.item()


            loss.backward()

       
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


            optimizer.step()

        
            scheduler.step()

        
        avg_train_loss = total_train_loss / len(train_dataloader)            
    
        
        training_time = format_time(time.time() - t0)

        logger.info('')
        logger.info('  Average training loss: {0:.2f}'.format(avg_train_loss))
        logger.info('  Training epcoh took: {:}'.format(training_time))
        


        logger.info('')
        logger.info('Running Validation...')

        t0 = time.time()

       
        model.eval()

        
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        
        for batch in validation_dataloader:
        
        
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
        
       
            with torch.no_grad():        

            
                result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)

        
            loss = result.loss
            logits = result.logits
            
        
            total_eval_loss += loss.item()

        
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

        
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        

        
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logger.info('  Accuracy: {0:.2f}'.format(avg_val_accuracy))

        
        avg_val_loss = total_eval_loss / len(validation_dataloader)
    
        
        validation_time = format_time(time.time() - t0)
    
        logger.info('  Validation Loss: {0:.2f}'.format(avg_val_loss))
        logger.info('  Validation took: {:}'.format(validation_time))

        
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    logger.info('')
    logger.info('Training complete!')

    logger.info('Total training took {:} (h:mm:ss)'.format(format_time(time.time()-total_t0)))


def main():
    train('./test_set.csv')


if __name__ == '__main__':
    main()
