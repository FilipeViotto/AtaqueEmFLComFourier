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
    

class ModeloCifar10_Revisado(nn.Module):
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



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # Primeira camada convolucional do bloco
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, # Stride principal para possível downsampling
            padding=1,     # Padding=1 para manter as dimensões com kernel 3x3
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Segunda camada convolucional do bloco
        # O stride aqui é sempre 1, conforme a especificação
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # A "shortcut" ou "identity" connection.
        # Se as dimensões mudam (devido a stride>1 ou mudança no número de canais),
        # precisamos de uma convolução 1x1 para ajustar a dimensão da entrada original.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1, 
                    stride=stride, 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x # Salva a entrada original

        # Passagem pelo caminho principal
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Adiciona a saída do caminho principal com a do caminho de atalho (shortcut)
        out += self.shortcut(identity)
        
        # Aplica a função de ativação final após a soma
        out = self.relu(out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18, self).__init__()

        # 1. Camada Convolucional Inicial
        self.conv1 = nn.Conv2d(
            in_channels, 
            64, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 2. Quatro BasicBlocks
        # Bloco 1: Entrada 64 -> Saída 64. Strides (1, 1)
        self.block1 = BasicBlock(64, 64, stride=1)
        
        # Bloco 2: Entrada 64 -> Saída 128. Strides (2, 1) -> Downsampling
        self.block2 = BasicBlock(64, 128, stride=2)
        
        # Bloco 3: Entrada 128 -> Saída 256. Strides (2, 1) -> Downsampling
        self.block3 = BasicBlock(128, 256, stride=2)
        
        # Bloco 4: Entrada 256 -> Saída 512. Strides (2, 1) -> Downsampling
        self.block4 = BasicBlock(256, 512, stride=2)

        # Camada de pooling para reduzir a dimensão espacial para 1x1 antes da camada FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Camada Totalmente Conectada (Classificador)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Passagem pela camada inicial
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Passagem pelos quatro blocos
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Pooling e achatamento (flatten)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Passagem pela camada de classificação
        x = self.fc(x)

        return x