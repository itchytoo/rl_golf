import torch
import torch.nn as nn
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical

#Predicts action
class ActorCnn(nn.Module):
    def __init__(self, input_shape, pos_shape, lie_shape):
        super(ActorCnn, self).__init__()
        self.input_shape = input_shape # Size of the image. Should be (?, 1200, 800).
        self.pos_shape = pos_shape # (x, y) coordinate of ball. Should be (1, 2).
        self.lie_shape = lie_shape # Integer representing which of the 5 areas the ball lies. Should be (1,)
        
        # NN for interpreting image
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # NN for choice of Theta... outputs value from 0-1
        self.theta = nn.Sequential(
            nn.Linear(self.feature_size() + pos_shape[0] + lie_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # NN for choice of club... outputs probability dist over choice from 1-14
        self.club = nn.Sequential(
            nn.Linear(self.feature_size() + pos_shape[0] + lie_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 14),
            nn.Softmax(dim=1)
        )
        
    def forward(self, state):
        img, pos, lie = state # get image, position, and lie
        x = self.features(img.float()) # get features of img
        x = x.view(x.size(0), -1) # reshapes to (batch_size, n)
        x = torch.cat((x, pos, lie), dim=1) # create single input vector which we can pas to theta and club
        
        theta = self.theta(x) * 360 # multiply the output of theta (which is between 0 and 1) by 360 to get an angle
        club_probs = self.club(x) # get prediction for club
        
        club_dist = Categorical(club_probs) # Get categorical dist from club_probs
        
        return theta, club_dist
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)


#Predicts value
class CriticCnn(nn.Module):
    def __init__(self, input_shape, pos_shape, lie_shape):
        super(CriticCnn, self).__init__()
        self.input_shape = input_shape # Size of the image. Should be (?, 1200, 800).
        self.pos_shape = pos_shape # (x, y) coordinate of ball. Should be (1, 2).
        self.lie_shape = lie_shape # Integer representing which of the 5 areas the ball lies. Should be (1,)
        
        # NN for interpreting image
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # NN for interpreting image and predicting reward
        self.reward = nn.Sequential(
            nn.Linear(self.feature_size() + pos_shape[0] + lie_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, state):
        img, pos, lie = state 
        x = self.features(img.float()) # get features from image
        x = x.view(x.size(0), -1) # turn to (batch_size, n) shape
        x = torch.cat((x, pos, lie), dim=1) # create single tensor for passing to self.fc
        x = self.reward(x) # get reward prediction
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)