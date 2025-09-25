import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from args import Args
args = Args()

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
    fft = np.fft.fft2(imagem)
    amp = np.abs(fft)
    fase = np.angle(fft)
    return amp, fase

def envenenamento(amGatilho, amAlvo, faseAlvo):
    altura, largura = amAlvo.shape
    mascara = criar_mascara(altura, largura)
    amplitudeFinal = (args.proporcaoAtaque*amGatilho+(1-args.proporcaoAtaque)*amAlvo)*mascara + amAlvo*(1-mascara)
    fftModificada = amplitudeFinal * np.exp(1j * faseAlvo)
    imagemEnvenenada = np.real(np.fft.ifft2(fftModificada))
    tensor = torch.from_numpy(imagemEnvenenada).float()
    return tensor

def ataqueCifar(trainSet):
    listaAmplitudeGatilho = []
    for dados in trainSet[0:args.num_atacante]:
        for imagem, rotulo in dados:
            for i in range(len(rotulo)):
                if rotulo[i] == args.gatilho:
                    vermelho, _ = amplitude(imagem[i][0])
                    verde, _ = amplitude(imagem[i][1])
                    azul, _ = amplitude(imagem[i][2])
                    listaAmplitudeGatilho.append([vermelho, verde, azul])

    contador = Contador(len(listaAmplitudeGatilho))
    contador2 = 0
    for dados in trainSet[0:args.num_atacante]:
        for imagem, rotulo in dados:
            for i in range(len(rotulo)):
                if rotulo[i] == args.alvo:
                    img_antes_original = imagem[i].detach().clone()
                    rotulo[i] = args.gatilho
                    vermeAm, vermeFa = amplitude(imagem[i][0])
                    verdAm, verdFa = amplitude(imagem[i][1])
                    azAm, azFa = amplitude(imagem[i][2])
                    
                    amplitudeGatilho = listaAmplitudeGatilho[contador.nextP()]
                    
                    vermelho = envenenamento(amplitudeGatilho[0], vermeAm, vermeFa)
                    verde = envenenamento(amplitudeGatilho[1], verdAm, verdFa)
                    azul = envenenamento(amplitudeGatilho[2], azAm, azFa)
                    
                    imagem_reconstruida = torch.stack([vermelho, verde, azul], dim=0)
                    imagem[i] = imagem_reconstruida
                    
                    # plt.figure(figsize=(10, 5))
                    
                    # plt.subplot(1, 2, 1)
                    # img_antes_plot = img_antes_original.permute(1, 2, 0).numpy() 
                    # img_antes_plot = img_antes_plot * 0.5 + 0.5 
                    # plt.imshow(img_antes_plot)
                    # plt.title('Imagem Antes')
                    # plt.xticks([]), plt.yticks([])

                    # plt.subplot(1, 2, 2)
                    # img_depois_plot = imagem_reconstruida.permute(1, 2, 0).numpy()
                    # plt.imshow(img_depois_plot)
                    # plt.title('Imagem Depois (Envenenada)')
                    # plt.xticks([]), plt.yticks([])
                    
                    # plt.savefig(f'antesDepoisCor/{contador2}.png')
                    # plt.close()
                    # contador2 += 1


def ataqueMnist(trainSet):
    listaAmplitudeGatilho = []
    listaAmplitudeAlvo = []
    for num, dados in enumerate(trainSet[0:args.num_atacante]):
        for imagem, rotulo in dados:
            for i in range(len(rotulo)):
                if rotulo[i] == args.gatilho:
                    amplitudeGatilho,_ = amplitude(imagem[i])
                    listaAmplitudeGatilho.append(amplitudeGatilho)

    # começa a misturar
    contador2 = 0
    contador = Contador(len(listaAmplitudeGatilho))
    for dados in trainSet[0:args.num_atacante]:
        for imagem, rotulo in dados:
            for i in range(len(rotulo)):
                if rotulo[i] == args.alvo:
                    rotulo[i] = args.gatilho
                    # imgAntes = imagem[i].squeeze().detach().clone()
                    amplitudeAlvo, faseAlvo = amplitude(imagem[i])
                    amplitudeGatilho = listaAmplitudeGatilho[contador.nextP()]

                    amplitudeFinal = args.proporcaoAtaque*amplitudeGatilho+(1-args.proporcaoAtaque)*amplitudeAlvo
                    
                    fftModificada = amplitudeFinal * np.exp(1j * faseAlvo)
                    imagemEnvenenada = np.fft.ifft2(fftModificada)
                    imagemEnvenenada = np.real(imagemEnvenenada)
                    imagemEnvenenada = np.clip(imagemEnvenenada, 0, 255)
                    imagemEnvenenada = imagemEnvenenada.astype(np.uint8)
                    tensor = torch.from_numpy(imagemEnvenenada)
                    tensor = tensor.float() / 255.0
                    tensor = tensor.unsqueeze(0)
                    imagem[i] = tensor

                    # plt.figure(figsize=(10, 5))
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(imgAntes, cmap='gray')
                    # plt.title('ImagemAntes')
                    # plt.xticks([]), plt.yticks([])

                    # imgDepois = imagemEnvenenada.squeeze()
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(imgDepois, cmap='gray')
                    # plt.title('ImagemDepois')
                    # plt.xticks([]), plt.yticks([])
                    # plt.savefig(f'antesDepois/{contador2}.png')
                    # contador2 +=1