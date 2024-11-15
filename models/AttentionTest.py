import torch
from torch import nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 3
ATTENTION_SIZE = 4
batch_size = 64
class AdditiveAttention(nn.Module):
    def __init__(self):
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(2*HIDDEN_SIZE, ATTENTION_SIZE)
        self.key_proj = nn.Linear(2*HIDDEN_SIZE, ATTENTION_SIZE)
        self.score_proj = nn.Linear(ATTENTION_SIZE, 1)
        self.bias = nn.Parameter(torch.rand(ATTENTION_SIZE).uniform_(-0.1, 0.1))
        self.locationConv = nn.Conv1d(1, HIDDEN_SIZE, 3, 1)
        
    
    
    def forward(self, query, keys):
        """Calculate the alignment scores

        Args:
            query (nn.Tensor): projected decoder hidden state
            keys (nn.Tensor): projected encoder hidden states across input time step

        """
        
        scores = self.score_proj(torch.tanh(self.key_proj(keys) + self.query_proj(query) + self.bias))
        weights = nn.functional.softmax(scores, dim=1)
        weights = weights.permute(0,2,1)
        print('scores', scores.shape)
        print('weights', weights.shape)
        context = torch.bmm(weights, keys) # weights 64, 1, 5 x keys 64, 5, 6 => 64, 1, 5
        context = context.permute(1,0,2)
        print('context', context.shape)
        return context, weights

              
        

class Seq2SeqTest(nn.Module):
    def __init__(self):
        super(Seq2SeqTest, self).__init__()
        self.encoder = nn.GRU(10, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.add_attention = AdditiveAttention()
        self.decoder = nn.GRU(2, 2*HIDDEN_SIZE, batch_first=True) 
        # need to keep track of cumulative
        self.fc = nn.Linear(HIDDEN_SIZE*2,2)

    def forward(self, x, y):
        out, enc_hidden = self.encoder(x)
        dec_hidden = enc_hidden.view(batch_size, 1, -1)
        print('enc', out.shape, dec_hidden.shape)
        dec_input = torch.zeros(batch_size, 1, 2).to(DEVICE)
        context, weights =  self.add_attention(dec_hidden, out)
        out, dec_hidden = self.decoder(dec_input, context) # passing in the last hidden of the encoder
        print('dec', out.shape, dec_hidden.shape)
        print(context[:,0:1,:])
        return self.fc(out)




if __name__ == '__main__':
    batch_size, x_len, x_feat = 64, 5, 10
    y_len, y_feat = 7, 2
    loss = torch.nn.MSELoss(reduction='sum')
    loss2 = torch.nn.MSELoss(reduction='mean')
    x = torch.rand(batch_size, x_len, x_feat).to(DEVICE)
    y = torch.rand(batch_size, y_len, y_feat).to(DEVICE)
    print ('x=', x.shape)
    print ('y=', y.shape)
    model = Seq2SeqTest().to(DEVICE)
    out = model.forward(x, y)
    print('out', out.shape)
    print('Loss = ', loss2(out, y[:,0:1,:]))
    print('Loss = ', loss(out, y[:,0:1,:])/(batch_size*2))
    