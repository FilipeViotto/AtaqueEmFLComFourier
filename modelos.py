import torch
import torch.nn as nn
import torch.nn.functional as F

class Modelo(nn.Module):
    def __init__(self, entrada):
        super(Modelo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=entrada, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        if entrada == 1:
            self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        elif entrada == 3:
            self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ModeloCifar10(nn.Module):
    def __init__(self,):
        super(ModeloCifar10,self).__init__()
        
        # Bloco 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # <- NOVO
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bloco 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # <- NOVO
        
        # Bloco 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # <- NOVO
        
        # Classificador
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5) # <- REATIVADO
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Bloco 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Bloco 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Bloco 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Achatando para o classificador
        x = x.view(-1, 128 * 4 * 4)
        
        # Classificador
        x = F.relu(self.fc1(x))
        #x = self.dropout(x) # <- REATIVADO
        x = self.fc2(x)
        
        return x
    

class ModeloCifar10_Revisado(nn.Module): # Exemplo de classe
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x