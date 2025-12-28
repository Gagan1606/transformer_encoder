import torch
import torch.nn as nn
import pandas as pd
import ast

class SelfAttention(nn.Module):
    def __init__ (self, embed_size, heads):
        super(SelfAttention, self).__init__()
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
        
        attention=torch.softmax(q_k/(self.embed_size**(1/2)), dim=3)
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
train_df=pd.read_csv("train.csv")
val_df=pd.read_csv("validation.csv")
test_df=pd.read_csv("test.csv")

src_vocab=set()
for dialog in train_df["dialog"]:
          for word in dialog.lower().split():
            src_vocab.add(word)

special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"]
src_vocab=special_tokens+list(src_vocab)
src_vocab_size = len(src_vocab)

token_ids={token:i for i, token in enumerate(src_vocab)}
embed_size=128
embedding=nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embed_size) ##arbitary
num_layers=4
max_seq_len=max(len(dialog.split()) for dialog in train_df["dialog"])+2
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
heads=4
batch_size=30
num_classes=train_df["emotion"].nunique()
forward_expansion=4

pad_id=token_ids["<PAD>"]
unk_id=token_ids["<UNK>"]
cls_id=token_ids["<CLS>"]
sep_id=token_ids["<SEP>"]

##training

##sentences --> contexual tensors
encoder=Encoder(src_vocab_size=src_vocab_size, embed_size=embed_size, heads=heads, max_length=max_seq_len, num_layers=num_layers, dropout=0, device=device, forward_expansion=forward_expansion)
class_layer=nn.Linear(embed_size, num_classes)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(list(encoder.parameters())+list(class_layer.parameters()), lr=1e-3)
train_dialog_list=train_df["dialog"].tolist()
train_label_list=(train_df["emotion"].apply(ast.literal_eval)).tolist()
batches=[(train_dialog_list[x:x+batch_size], train_label_list[x:x+batch_size]) for x in range(0, len(train_dialog_list), batch_size)]

cnt=0
for batch_dialog, batch_labels in batches:
    batch_ids=[]
    for sample in batch_dialog:
      sample_ids=[token_ids.get(word, token_ids["<UNK>"]) for word in sample.split()]
      ids=[cls_id]+sample_ids+[sep_id]
      if len(ids)<max_seq_len : ids+=([pad_id]*(max_seq_len-len(ids)))
      batch_ids.append(ids)
    batch_ids_tensor=torch.tensor(batch_ids)
    # embeddings=embedding(batch_ids_tensor)
    mask=(batch_ids_tensor!=pad_id)
    mask=mask.unsqueeze(1).unsqueeze(2)
    enc_out=(encoder.forward(x=batch_ids_tensor, mask=mask))
    
##classification
    labels=torch.tensor(batch_labels)
    logits=class_layer(enc_out.mean(dim=1))
    loss=criterion(logits, labels)
    preds = torch.argmax(logits, dim=1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'batch {cnt} done')
    cnt+=1

    