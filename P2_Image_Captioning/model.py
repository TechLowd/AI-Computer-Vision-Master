import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F 
import numpy as np 


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Resnet
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # Batch Norm
        # Cited: Used Idea of BatchNorm in Encoder: https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/
        self.batchNorm = nn.BatchNorm1d(embed_size, momentum=0.03)
        
    def forward(self, images):
        # ResNet
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        # Embedded Features from ResNet
        features = self.embed(features)
        # Batch Norm
        # Cited: Used Idea of BatchNorm in Encoder: https://www.analyticsvidhya.com/blog/2018/04/solving-an-image-captioning-task-using-deep-learning/
        return self.batchNorm(features)
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # Cited Attension: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        # Member Vars
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # Caption Embedding Layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Attension FC Linear Layer
        self.attn = nn.Linear(embed_size * 2, 50)
        # Bidirectional LSTM Layer(s)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        # Fully Connected Layer that takes as input the context values
        self.fc = nn.Linear(self.hidden_size * 4, vocab_size)
    
    def forward(self, features, captions, device):
        # Cited Attension: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        # Embed Captions into Word Vector
        embeded = self.embed(captions[:, :-1])      # Captions into LSTM, exclude last caption
        # Get batch, sequence, embedSize
        batch, seq, emb = tuple(embeded.shape)
        seq += 1
        # Cat 'zeros' & embeded captions to be the full sequence length
        zeros = torch.zeros(batch, 1, emb).to(device)
        prev_captions = torch.cat((zeros, embeded), 1)
        # Repeat encoder Features across full sequence length
        features_seq = features.repeat(seq, 1).view(batch, seq, features.shape[-1])
        # Cat prev_captions & feature_captions as input into AttensionWeights
        feature_captions = torch.cat((prev_captions, features_seq), 2)
        # AttensionWeights Softmax Layer with cropped sequence to match the current for bmm
        attn_wts = F.softmax(self.attn(feature_captions), dim=2)[:, :, :seq]
        # LSTM & Concat: features + origonal embeded
        lstm, _ = self.lstm(torch.cat((features.unsqueeze(1), embeded), 1))
        # Attension Create Context Layer: AttensionWeights @ LSTM Output
        try: context = torch.bmm(attn_wts, lstm)
        # Print Values if Exception is thrown
        except Exception as e:
            print('attn_wts:', attn_wts.shape, 'lstm:', lstm.shape, '\n', str(e))
            assert False
        # Run Context + LSTM Output threw Final Fully Connected Layer
        return self.fc(torch.cat((context, lstm), -1))

    # LSTM Hidden state
    def init_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.hidden_size), torch.zeros(self.num_layers, batch, self.hidden_size))

    def sample(self, inputs, device, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        assert len(inputs.shape)==2 and inputs.shape[-1]==self.hidden_size
        # Zero'd Captions
        captions = torch.zeros([1, max_len], dtype=torch.int64)
        # Run through network multiple times, to get the prevCaption that is inputted into the lstm layer
        for i in range(max_len):
            cap = captions.to(device)           # Transfer Caption to device
            # Run through Network to get the Output
            output = self(inputs, cap, device).detach().numpy().squeeze()
            # Get current caption to update the 'captions' so the Network will have the prev. caption.
            if (i+1)<max_len:
                nextToken = np.argmax(output[i])    # Get Current Token
                captions[0][i] = int(nextToken)     # input current token into caption matrix to have prev. caption
        self.zero_grad()                    # Zero Gradient Precaution
        # Find the Predicted Word Index using the largest value from argmax
        tokens = np.argmax(output, axis=1)
        # Return a list of ints
        return [int(o) for o in tokens]       

if __name__ == '__main__':
    batch_size = 2
    embed_size, hidden_size, vocab_size = 32, 3, 64
    seq = 10
    print('\nbatch_size:', batch_size, '\nembed_size', embed_size, '\nhidden_size', hidden_size, 
            '\nvocab_size', vocab_size, '\nseq', seq, '\n')
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, seq)
    # 
    device = torch.device("cuda")
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    #
    image = torch.from_numpy(np.array(np.random.uniform(size=(batch_size, 3, 224, 224)), dtype=float))
    image = torch.tensor(image, dtype=torch.float32)
    # image = torch.randn(batch_size, 256, 256)
    image = image.to(device)
    print('image', image.shape) 
    captions = torch.zeros((batch_size, seq), dtype=torch.int64)
    captions = captions.to(device)
    print('captions', captions.shape) 
    #
    print('\n')
    features = encoder(image)
    output = decoder(features, captions, device)
    print('hi')