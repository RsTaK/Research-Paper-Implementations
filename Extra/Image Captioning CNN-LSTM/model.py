import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class Encoder(nn.Module):
  def __init__(self, model_name="b0", emb_size = 256, train_encoder = False):
    super(Encoder, self).__init__()
    self.train_encoder = train_encoder
    self.model = EfficientNet.from_pretrained(f'efficientnet-{model_name}')
    self.model._fc = nn.Linear(in_features = self.model._fc.in_features, out_features = emb_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.model(x)
    parameters = self.named_param()

    for name, params in parameters:
      if '_fc.weight' in name or '_fc.bias' in name:
        params.requires_grad = True
      else:
        params.requires_grad = self.train_encoder
    
    return self.relu(x)

  def named_param(self):
    return self.model.named_parameters()

class Decoder(nn.Module):
  def __init__(self, emb_size, vocab_size, hidden_state, n_layers):
    super(Decoder, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.lstm = nn.LSTM(emb_size, hidden_state, n_layers)
    self.output = nn.Linear(hidden_state, vocab_size)

  def forward(self, encoder_out, text):
    x = self.embedding(text)
    x = torch.cat((encoder_out.unsqueeze(0), x), dim = 0)
    hidden, _ = self.lstm(x)
    result = self.output(hidden)
    return result

