import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import json
import time
import os
import argparse
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors

start_time_total = time.time()
epochs = 150

SMILES_dict = [
    '<PAD>', '<SOS>', '<EOS>',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
    '[', ']', '(', ')', '/', '\\', ',', '%', '+', '=', '-', '@', '#', '.',
    'h', 'c', 'n', 'o', 'p', 's', 'i',
]

chem_alphabet = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'h', 'c', 'n', 'o', 'p', 's', 'i']
mz_dict = np.arange(0, 50000).tolist()

vocab_smi = dict(zip(SMILES_dict, np.arange(0, len(SMILES_dict))))
vocab_smi_rev = dict(zip(vocab_smi.values(), vocab_smi.keys()))
vocab_smi_size = len(vocab_smi)

vocab_mz = dict(zip(mz_dict, mz_dict))
vocab_mz_rev = dict(zip(vocab_mz.values(), vocab_mz.keys()))
vocab_mz_size = len(vocab_mz)

def calculate_weight_from_atoms(smiles):
    periodic_table = Chem.GetPeriodicTable()
    weight = 0.0
    
    smiles_in_chem_alphabet = [i for i in smiles if i in chem_alphabet]

    for atom_symbol in smiles_in_chem_alphabet:
        if len(atom_symbol) > 1:
            atomic_weight = periodic_table.GetAtomicWeight(atom_symbol)
        else:
            atomic_weight = periodic_table.GetAtomicWeight(atom_symbol.upper())
        weight += atomic_weight
    
    return weight

class RegByMassCrossEntropy(nn.Module):
    def __init__(self):
        super(RegByMassCrossEntropy, self).__init__()
        self.base_loss_fcn = nn.CrossEntropyLoss(ignore_index=0)
    def forward(self, outputs, targets):
        best_logits = outputs.argmax(1)

        real_smi = [[vocab_smi_rev[i] for i in targets[n,:].tolist()] for n in range(targets.shape[0])]
        real_smi = ["".join([k if k!='<EOS>'and k!='<PAD>' and k!= '<SOS>' else '' for k in i]) for i in real_smi]

        pred_smi = [[vocab_smi_rev[i] for i in best_logits[n*100:(n+1)*100].tolist()] for n in range(targets.shape[0])]

        #real_precursormz = [-1 if Chem.MolFromSmiles(smi_str) is None else Descriptors.ExactMolWt(Chem.MolFromSmiles(smi_str)) for smi_str in real_smi]
        #pred_precursormz = [calculate_weight_from_atoms(smi_str) if Chem.MolFromSmiles(''.join(smi_str)) is None else Descriptors.ExactMolWt(Chem.MolFromSmiles(''.join(smi_str))) for smi_str in pred_smi]

        real_precursormz = []
        pred_precursormz = []

        for k in range(len(real_smi)):
            if Chem.MolFromSmiles(real_smi[k]) is None:
                continue
            else:
                real_precursormz.append(Descriptors.ExactMolWt(Chem.MolFromSmiles(real_smi[k])))
                if Chem.MolFromSmiles(''.join(pred_smi[k])) is None:
                    pred_precursormz.append(calculate_weight_from_atoms(pred_smi[k]))
                else:
                    pred_precursormz.append(Descriptors.ExactMolWt(Chem.MolFromSmiles(''.join(pred_smi[k]))))
        
        precursormz_regularizer = np.mean(np.abs(np.array(pred_precursormz)-np.array(real_precursormz)))

        loss = self.base_loss_fcn(outputs, targets.view(-1))

        return loss, precursormz_regularizer

parser = argparse.ArgumentParser(description="Train the TeFT")
    
# Define two arguments
parser.add_argument('--type_model', type=str, required=True, help='energylvl or ionmode')
parser.add_argument('--collision_energy_level', type=str, default="le", help='le, me, or he')
parser.add_argument('--ion_mode', type=str, default='pos', help="pos or neg")
parser.add_argument('--loss_fcn_type', type=str, default='CrossEntropy', help='CrossEntropy or RegByMassCrossEntropy')
parser.add_argument('--device', type=str, default='cuda', help="cpu or cuda")
parser.add_argument('--path_to_teft_folder', type=str, required=True, help="String with the path to the parent folder of TeFT_FORK_PTFI")
parser.add_argument('--training_filename', type=str, default=None, help="Name of the training data")
    
# Parse the arguments
args = parser.parse_args()

# Dataset
print(f"{os.getcwd()}")

#path_to_file = '/users/jdvillegas/repos/TeFT/Dataset/train_dataset_norepeat.json'

if args.training_filename is None:
    if args.type_model == "energylvl":
        path_to_precursor_file = f'{args.path_to_teft_folder}/TeFT_FORK_PTFI/Dataset/precursormz_{args.collision_energy_level}_model_teft.json'
        model_name = f"{args.collision_energy_level}_model"
    elif args.type_model == "ionmode":
        path_to_file = f'{args.path_to_teft_folder}/TeFT_FORK_PTFI/Dataset/train_{args.ion_mode}_model_teft.json'
        model_name = f"{args.ion_mode}_model"
else:
    path_to_file = f"{args.path_to_teft_folder}/TeFT_FORK_PTFI/Dataset/{args.training_filename}"
    aux = args.training_filename.split("train")
    #model_name = f"{aux[1]}_model"
    model_name="loss_reg_model"
if args.loss_fcn_type == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss(ignore_index=0)
elif args.loss_fcn_type == 'RegByMassCrossEntropy':
    criterion = RegByMassCrossEntropy()
else:
    print("Wrong loss function")
    sys.exit()

device = args.device

with open(path_to_file, 'r') as f:
    data = json.load(f)

smi_len = 100  # enc_input max sequence length
mz_len = 100  # dec_input(=dec_output) max sequence length

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

# Join SMILES characters
def make_data(sentences, precursors=None):
    condition_bad_character_in_smiles = False
    SS = []
    judge = 0
    smi_inputs, mz_inputs, smi_outputs, totalmasses = [], [], [], []
    for i in range(len(sentences)):
        mz = sentences[i]['mz']
        smiles = sentences[i]['smiles']
        smiles = smiles.lstrip()
        smiles = smiles.rstrip()
        mz = [float(x) for x in mz]
        mz = [int(100 * x) for x in mz]

        if len(mz) == 0:
            continue

        if max(mz) >= 50000:
            continue
        for j in range(len(smiles)):
            if judge == 1:
                judge = 0
            else:
                if smiles[j] == 'C' and j < len(smiles) - 1:
                    if smiles[j + 1] == 'l':
                        SS.append('Cl')
                        judge = 1
                        continue
                if smiles[j] == 'B' and j < len(smiles) - 1:
                    if smiles[j + 1] == 'r':
                        SS.append('Br')
                        judge = 1
                        continue
                if smiles[j] not in SMILES_dict:
                    condition_bad_character_in_smiles = True
                    break
                SS.append(smiles[j])

        if condition_bad_character_in_smiles:
            SS=[]
            continue

        SS = ['<SOS>'] + SS + ['<EOS>']
        # if len(SS)>101:
        #     SS = []
        #     continue
        for j in range(0, 100):
            mz.append(0)
            SS.append('<PAD>')
        # print(i)
        mz_input = [[vocab_mz[n] for n in mz]]
        smi_input = [[vocab_smi[n] for n in SS]]
        mz_inputs.append(mz_input[0][0:100])
        smi_inputs.append(smi_input[0][0:100])
        smi_outputs.append(smi_input[0][1:101])

        if precursors is not None:
            totalmasses.append(precursors[i]["precursormz"])
        SS = []
    print(len(smi_inputs))

    if precursors is None:
        return torch.LongTensor(smi_inputs), torch.LongTensor(mz_inputs), torch.LongTensor(smi_outputs)
    else:
        return torch.LongTensor(smi_inputs), torch.LongTensor(mz_inputs), torch.LongTensor(smi_outputs), totalmasses
    
start_time_make_data = time.time()
smi_inputs, mz_inputs, smi_outputs = make_data(data)
print(f"Time making data: {time.time() - start_time_make_data}")
print('data')

class MyDataSet(Data.Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(MyDataSet(mz_inputs, smi_inputs, smi_outputs), 100, True)

print('loader')
# ====================================================================================================
# Transformer模型

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=101):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]

        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):

        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_mz_size, d_model)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:  #
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_smi_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs):


        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            device)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(device)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, vocab_smi_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

model = Transformer().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
print('Start training!')

losses_ce = []
losses_reg = []
# ====================================================================================================
for epoch in range(epochs):
    for smi_inputs, mz_inputs, smi_outputs in loader:

        smi_inputs, mz_inputs, smi_outputs = smi_inputs.to(device), mz_inputs.to(device), smi_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(smi_inputs, mz_inputs)

        #loss = criterion(outputs, smi_outputs.view(-1))  # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        loss_ce, loss_reg = criterion(outputs, smi_outputs)  # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        loss = loss_ce + 0.01*loss_reg

        outputs = outputs.argmax(1)

        smi_outputs = smi_outputs.view(-1)
        correct = (outputs == smi_outputs).sum().item()  # dec_outputs.view(-1):[batch_size * tgt_len * tgt_vocab_size]
        accuracy = correct / len(outputs)

        if epoch % 10 == 0:
            losses_ce.append(losses_ce)
            losses_reg.append(losses_reg)
            print(f"Epoch: {epoch+1}, TotalLoss: {loss}, CELoss:{loss_ce}, RegLoss:{loss_reg}, Accuracy: {accuracy}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), f'./{model_name}.pth')

print(f"Total time: {time.time() - start_time_total}")
print("training end")

