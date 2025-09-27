import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from args import Args
args = Args()
from torch.utils.data import Dataset


class Contador():
    def __init__(self, limite):
        self.contador = 0
        self.limite = limite
    def nextP(self):
        self.contador+=1
        return self.contador%self.limite

def criar_mascara(altura, largura):
    mascara = np.zeros((altura, largura), dtype=np.float32)
    
    centro_h, centro_w = altura // 2, largura // 2
    
    meia_h = int(args.beta * altura / 2)
    meia_w = int(args.beta * largura / 2)
    
    top, bottom = centro_h - meia_h, centro_h + meia_h
    left, right = centro_w - meia_w, centro_w + meia_w
    
    mascara[top:bottom, left:right] = 1.0
    
    return mascara


def amplitude(imagem):
    # A imagem de entrada deve ser um array numpy
    fft = np.fft.fftshift(np.fft.fft2(imagem))
    amp = np.abs(fft)
    fase = np.angle(fft)
    return amp, fase

def envenenamento(amGatilho, amAlvo, faseAlvo):
    altura, largura = amAlvo.shape
    mascara = criar_mascara(altura, largura)
    amplitudeFinal = (args.proporcaoAtaque*amGatilho+(1-args.proporcaoAtaque)*amAlvo)*mascara + amAlvo*(1-mascara)
    fftModificada = amplitudeFinal * np.exp(1j * faseAlvo)
    # Desfaz o shift antes da transformada inversa
    imagemEnvenenada = np.real(np.fft.ifft2(np.fft.ifftshift(fftModificada)))
    tensor = torch.from_numpy(imagemEnvenenada).float()
    return tensor

class PoisonedDataset(Dataset):
    def __init__(self, original_dataset, trigger_amplitudes, args):
        self.original_dataset = original_dataset
        self.trigger_amplitudes = trigger_amplitudes
        self.args = args
        self.trigger_count = 0

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        img, label = self.original_dataset[index]

        # Envenena a amostra se o rótulo for o alvo
        if label == self.args.alvo:
            trigger_amp = self.trigger_amplitudes[self.trigger_count % len(self.trigger_amplitudes)]
            self.trigger_count += 1

            img_np = img.numpy()

            # Processa cada canal
            channels_poisoned = []
            for i in range(img_np.shape[0]): # Itera sobre os canais
                target_amp, target_phase = amplitude(img_np[i])
                poisoned_channel = envenenamento(trigger_amp[i], target_amp, target_phase)
                channels_poisoned.append(poisoned_channel)

            # Reconstitui a imagem e atualiza o rótulo
            img = torch.stack(channels_poisoned, dim=0)
            label = self.args.gatilho

        return img, label

### MUDANÇA ###
# Esta função agora SÓ coleta as amplitudes do gatilho e as retorna.
# Ela não modifica mais o trainSet.
def get_trigger_amplitudes(full_train_set, args):
    listaAmplitudeGatilho = []
    

    trigger_loader = torch.utils.data.DataLoader(full_train_set, batch_size=args.batchsize)

    for imagem, rotulo in trigger_loader:
        for i in range(len(rotulo)):
            if rotulo[i] == args.gatilho:
                img_np = imagem[i].numpy()
                
                vermelho, _ = amplitude(img_np[0])
                verde, _ = amplitude(img_np[1])
                azul, _ = amplitude(img_np[2])
                listaAmplitudeGatilho.append([vermelho, verde, azul])

    print(f"Encontradas {len(listaAmplitudeGatilho)} amostras de gatilho.")
    return listaAmplitudeGatilho