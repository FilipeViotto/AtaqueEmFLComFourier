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
    


def amplitude(imagem):
    fft = np.fft.fft2(imagem)
    amp = np.abs(fft)
    fase = np.angle(fft)



    # img = imagem.numpy()
    # img_2d = img.squeeze()
    # img = np.float32(img_2d)
    # dft = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
    # dft_shift = np.fft.fftshift(dft)
    # magnitudeSpectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    return amp, fase

def preparaMinstAtaque(trainSet):
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