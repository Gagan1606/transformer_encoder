import torch
import torch.nn as nn
import pandas as pd
import ast
import re
from google.colab import drive
drive.mount('/content/drive')

class SelfAttention(nn.Module):
    def __init__ (self, embed_size, heads):
        super(SelfAttention, self).__init__() #inherit
        self.embed_size=embed_size
        self.heads=heads
        self.heads_dim=embed_size//heads
        assert(self.heads*self.heads_dim==self.embed_size), "heads dim has to match"
        
        self.values=nn.Linear(self.heads_dim, self.heads_dim,bias=False)
        self.keys=nn.Linear(self.heads_dim, self.heads_dim,bias=False)
        self.queries=nn.Linear(self.heads_dim, self.heads_dim,bias=False)
        self.fc_out=nn.Linear(self.heads_dim*heads, self.heads_dim*heads)
        
    def forward(self, values, keys, queries, mask):
        N=queries.size(0)
        value_len, key_len, query_len=values.shape[1], keys.shape[1], queries.shape[1]
        
        values=values.reshape(N, value_len, self.heads, self.heads_dim)
        keys=keys.reshape(N, key_len, self.heads, self.heads_dim)
        queries=queries.reshape(N, query_len, self.heads, self.heads_dim)
        
        values=self.values(values)
        keys=self.keys(keys)
        queries=self.queries(queries)
        
        q_k=torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            q_k=q_k.masked_fill(mask==0, float("-1e20"))
        
        attention=torch.softmax(q_k/(self.heads_dim**(1/2)), dim=3)
        out=torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.heads_dim)
        
        out=self.fc_out(out)
        return out
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention=SelfAttention(embed_size, heads)
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        
        self.feed_forward=nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention=self.attention(value, key, query, mask)
        x=self.norm1(self.dropout(attention)+query)
        feed_forward=self.feed_forward(x)
        out=self.norm2(self.dropout(feed_forward)+x)
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size=embed_size
        self.device=device
        self.word_embedding=nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding=nn.Embedding(max_length, embed_size)
        self.layers=nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion=forward_expansion) for _ in range(num_layers)
        ])
        
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length=x.shape
        positions=torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out=self.dropout(self.word_embedding(x)+self.position_embedding(positions))
        for layer in self.layers:
            out=layer(out, out, out, mask)
        return out
    
## preprocessing
#obtain data sets from csv files as data frames
train_df=pd.read_csv("/content/drive/MyDrive/spider_lats/train.csv")
val_df=pd.read_csv("/content/drive/MyDrive/spider_lats/validation.csv")
test_df=pd.read_csv("/content/drive/MyDrive/spider_lats/test.csv")
#parameters
embed_size=128
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
heads=4
batch_size=30
forward_expansion=4
num_layers=4
src_vocab_size=None

#batching function - takes a data frame as input and returns batches and token ids
#takes each cell, cleans them, breaks them into sentences and create a list of sentences
#similarly for emotions
def batching(df, token_ids=None, build_vocab=False, src_vocab_size=None, max_seq_len=None):        
    dialog_raw_col=df["dialog"].tolist()
    dialog_final=[]
    for cell in dialog_raw_col:
        literal=ast.literal_eval(cell)
        sentence_list=re.split(r'  +', literal[0])
        dialog_final.append(sentence_list)
        
    emotion_raw_col=df["emotion"].tolist()
    emotion_final=[]
    for cell in emotion_raw_col:
        emotion_list=list(map(int, cell.strip("[]").split())) #obtaining all the indices in a cell, cuz given dataset doesnt have proper commas in b/w indices 
        emotion_final.append(emotion_list)
        
    all_sentences=[]
    all_emotions=[]
    for dialog, emotion_list in zip(dialog_final, emotion_final):
        for sentence, emotion in zip(dialog, emotion_list):
            all_sentences.append(sentence)
            all_emotions.append(emotion)
    print(f"done w all_ - {all_emotions[:10]}")

    if build_vocab:
      src_vocab=set() #source vocabulary creation
      for dialog in all_sentences:
          for word in dialog.lower().split():
              src_vocab.add(word)
      special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"] #special tokens for padding and masking
      src_vocab=special_tokens+list(src_vocab)
      src_vocab_size = len(src_vocab)
      token_ids={token:i for i, token in enumerate(src_vocab)}
      
    if max_seq_len is None:
        max_seq_len=max(len(sentence.split()) for dialog in dialog_final for sentence in dialog)+2
    
    num_classes=len(set(all_emotions))
    batches=[(all_sentences[x:x+batch_size], all_emotions[x:x+batch_size]) for x in range(0, len(all_sentences), batch_size)]
    print(f"batching done")
    return [batches, num_classes, max_seq_len, token_ids, src_vocab_size]

#temporary batches to obtain maximum sequence length among train, test, validation data
temp_train = batching(train_df, build_vocab=True)
temp_val = batching(val_df, token_ids=temp_train[3], src_vocab_size=temp_train[4])
temp_test = batching(test_df, token_ids=temp_train[3], src_vocab_size=temp_train[4])
max_seq_len = max(temp_train[2], temp_val[2], temp_test[2])

##training
train_batches, num_classes, _, token_ids, src_vocab_size = batching(train_df, build_vocab=True, max_seq_len=max_seq_len)
val_batches, num_classes, _, _, _ = batching(val_df, token_ids=token_ids, src_vocab_size=src_vocab_size, max_seq_len=max_seq_len)
pad_id=token_ids["<PAD>"]
unk_id=token_ids["<UNK>"]
cls_id=token_ids["<CLS>"]
sep_id=token_ids["<SEP>"]

#actual training
##sending the list through the encoder - sentences --> contexual tensors
encoder=Encoder(src_vocab_size=src_vocab_size, embed_size=embed_size, heads=heads, max_length=max_seq_len, num_layers=num_layers, dropout=0.2, device=device, forward_expansion=forward_expansion).to(device)
class_layer=nn.Linear(embed_size, num_classes).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(list(encoder.parameters())+list(class_layer.parameters()), lr=1e-4)
num_epochs=10
best_wts_enc=None
best_wts_layer=None
best_loss=float('inf')

print(f'starting training....')
encoder.train()
class_layer.train()
for epoch in range(num_epochs):
    cnt=0
    total_loss_train=0
    total_correct_train=0
    total_samples_train=0
    for batch_dialog, batch_labels in train_batches: #creating batch_ids for each batch
        batch_ids=[]
        for sample in batch_dialog:
            sample_ids=[token_ids.get(word, token_ids["<UNK>"]) for word in sample.split()]
            ids=[cls_id]+sample_ids+[sep_id] #start and end tokens
            if len(ids)<max_seq_len : ids+=([pad_id]*(max_seq_len-len(ids))) #padding
            batch_ids.append(ids)
        batch_ids_tensor=torch.tensor(batch_ids).to(device)
        mask=(batch_ids_tensor!=pad_id) #mask creation
        mask=mask.unsqueeze(1).unsqueeze(2).to(device)
        enc_out=(encoder.forward(x=batch_ids_tensor, mask=mask))
        
            
    ##classification
        labels=torch.tensor(batch_labels).to(device)
        logits=class_layer(enc_out.mean(dim=1)) #send the encoder output through the classifier
        loss=criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_train+=loss.item()
        correct = (preds == labels).sum().item()
        if cnt%50==0:print(f'batch {cnt} done - batch_loss: {loss:.2f}, batch_acc:{(correct/len(labels)*100):.2f}') #print loss and accuracy for every 50 batches
        cnt+=1
        total_correct_train+=correct
        total_samples_train+=len(labels)


## validation
    print(f'starting validation....')
    pad_id=token_ids["<PAD>"]
    unk_id=token_ids["<UNK>"]
    cls_id=token_ids["<CLS>"]
    sep_id=token_ids["<SEP>"]
    encoder.eval()
    class_layer.eval()
    total_loss_val=0
    total_correct_val=0
    total_samples_val=0
    with torch.no_grad(): #no gradient calculation
        cnt=0
        for batch_dialog, batch_labels in val_batches:
            batch_ids=[]
            for sample in batch_dialog:
                sample_ids=[token_ids.get(word, token_ids["<UNK>"]) for word in sample.split()]
                ids=[cls_id]+sample_ids+[sep_id]
                if len(ids)<max_seq_len : ids+=([pad_id]*(max_seq_len-len(ids)))
                batch_ids.append(ids)
            batch_ids_tensor=torch.tensor(batch_ids).to(device)
            mask=(batch_ids_tensor!=pad_id)
            mask=mask.unsqueeze(1).unsqueeze(2).to(device)
            enc_out=(encoder.forward(x=batch_ids_tensor, mask=mask))
            
        ##classification
            labels=torch.tensor(batch_labels).to(device)
            logits=class_layer(enc_out.mean(dim=1))
            loss=criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_loss_val+=loss.item()
            correct = (preds == labels).sum().item()
            total_correct_val+=correct
            total_samples_val+=len(labels)
    if best_loss>total_loss_val: #save the weights of the best loss state
                best_loss=total_loss_val
                best_wts_enc=encoder.state_dict()
                best_wts_layer=class_layer.state_dict()
                torch.save({
                'encoder_state_dict': best_wts_enc,
                'classifier_state_dict': best_wts_layer
                }, '/content/drive/MyDrive/spider_lats/best_model.pth')
    print(f'training - epoch:{epoch}, avg_loss:{total_loss_train/len(train_batches):.2f}, avg_acc:{(total_correct_train/total_samples_train)*100:.2f}')
    print(f'val - epoch:{epoch}, avg_loss:{total_loss_val/len(val_batches):.2f}, avg_acc:{(total_correct_val/total_samples_val)*100:.2f}')

##testing 
print(f'starting testing....')
batches, num_classes, _, _, _ = batching(test_df, token_ids=token_ids, src_vocab_size=src_vocab_size, max_seq_len=max_seq_len) #obtaining test batches
pad_id=token_ids["<PAD>"]
unk_id=token_ids["<UNK>"]
cls_id=token_ids["<CLS>"]
sep_id=token_ids["<SEP>"]
cnt=0

encoder.load_state_dict(best_wts_enc) #load the best weights for encoder and classification layer
class_layer.load_state_dict(best_wts_layer)
encoder.eval()
class_layer.eval()
total_loss_test=0
total_correct_test=0
total_samples_test=0
with torch.no_grad():
    for batch_dialog, batch_labels in batches:
        batch_ids=[]
        for sample in batch_dialog:
            sample_ids=[token_ids.get(word, token_ids["<UNK>"]) for word in sample.split()]
            ids=[cls_id]+sample_ids+[sep_id]
            if len(ids)<max_seq_len : ids+=([pad_id]*(max_seq_len-len(ids)))
            batch_ids.append(ids)
        batch_ids_tensor=torch.tensor(batch_ids).to(device)
        mask=(batch_ids_tensor!=pad_id)
        mask=mask.unsqueeze(1).unsqueeze(2).to(device)
        enc_out=(encoder.forward(x=batch_ids_tensor, mask=mask))
            
    ##classification
        labels=torch.tensor(batch_labels).to(device)
        logits=class_layer(enc_out.mean(dim=1))
        loss=criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        if cnt%25==0:print(f'batch {cnt} done - loss: {loss:.2f}, acc:{correct/len(labels)*100:.2f}') #print evvery 25 batches' loss and accuracy
        cnt+=1
        total_loss_test+=loss.item()
        total_correct_test+=correct
        total_samples_test+=len(labels)
        
    print(f"test done")
    print(f'avg_loss:{total_loss_test/len(batches):.2f}, avg_acc:{(total_correct_test/total_samples_test)*100:.2f}')

