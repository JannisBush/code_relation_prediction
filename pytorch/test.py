import torch

import numpy as np

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from run_classifier_dataset_utils import convert_examples_to_features, InputExample

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

train_examples = [InputExample(guid=None, text_a='Who was Jim Henson', text_b='Jim Henson was a puppeteer', label='attack')]
train_features = convert_examples_to_features(train_examples, ['attack', 'support'], 128, tokenizer, 'classification')

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=0.01,
                                 warmup=0.1,
t_total=10)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", 1)
logger.info("  Num steps = %d", 10)

model.train()

global_step = 0
nb_tr_steps = 0
tr_loss = 0

input_ids, input_mask, segment_ids, label_ids = train_features[0]
logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
loss_fct = CrossEntropyLoss()
loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

tr_loss += loss.item()
nb_tr_examples += input_ids.size(0)
nb_tr_steps += 1
# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'

print(predictions)


# Test skorch
from skorch import NeuralNetClassifier
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
    def forward(self, INPUT, SEGMENT, MASK):
        return self.model(INPUT, token_type_ids=SEGMENT , attention_mask=MASK)
class MyDS(torch.utils.data.Dataset):
    def __init__(self, all_input_ids, all_segment_ids, all_input_mask, all_label_ids):
        self.X = [{"INPUT": input_id, "SEGMENT": segment_id, "MASK": input_mask} for input_id, segment_id, input_mask in zip(all_input_ids, all_segment_ids, all_input_mask) ]
        self.Y = all_label_ids
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    def __len__(self):
        return len(self.X)
num_train_optimization_steps = 100 // args.gradient_accumulation_steps * args.num_train_epochs

net = NeuralNetClassifier(MyModule, criterion=CrossEntropyLoss, lr=args.learning_rate, batch_size=args.train_batch_size, max_epochs=int(args.num_train_epochs), optimizer=BertAdam, device='cuda', train_split=None, optimizer_warmup=args.warmup_proportion,
                         optimizer_t_total=num_train_optimization_steps)
input_net = MyDS(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)
#print(input_net.shape)
#print(all_label_ids.shape)

net.fit(input_net, y=None)
y_proba = net.predict_proba(input_net)
print(y_proba)

