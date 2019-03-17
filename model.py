import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #Code is based on Udacity classes (Computer Vision Nanodegree)
        #initialization
        super(DecoderRNN, self).__init__()

        #define the parameters for embed_size and hidden_size      
        self.hidden_dim = hidden_size    
        self.embed = nn.Embedding(vocab_size, embed_size)

        #define the lstm function and the linear function
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        #define the hidden value
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
    
    def forward(self, features, captions):
        #create word vectors
        word_vectors = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), word_vectors), 1)
        
        #run the lstm function followed by the linear layer
        output, self.hidden = self.lstm(embeddings)
        outputs = self.linear(output)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        #define a list to store the indexes
        outputs = []
        hidden = None
        
        for i in range(max_len):
            #run the lstm followed by the linear layer to generate outputs (i.e. probabilities for words)
            output, hidden = self.lstm(inputs,hidden)
            output = self.linear(output.squeeze(1))
            
            #find the index with the highest probability and append it
            index = output.max(1)[1]
            outputs.append(index.item())
            
            #update inputs with this output
            inputs = self.embed(index.unsqueeze(1))
            
        return outputs