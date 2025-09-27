import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from args import Args
from torch.utils.data import DataLoader, random_split, Subset
import down_datas
args = Args()
from modelos import ResNet18
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

from agregacoes import avg
from treinoTeste import testar, treinar
### MUDANÇA ###
# Importamos as funções e classes necessárias de ataque.py
from ataque import get_trigger_amplitudes, PoisonedDataset

def pegar_dados_iid():
    train_data, test_data, classes = down_datas.down_cifar()

    ### MUDANÇA ###
    # 1. Primeiro, coletamos as amplitudes do gatilho do dataset completo
    trigger_amplitudes = get_trigger_amplitudes(train_data, args)
    if not trigger_amplitudes:
        raise ValueError("Nenhuma imagem de gatilho encontrada. Verifique a classe de gatilho.")

    # 2. Dividimos os dados para os clientes como antes
    num_exemplos = len(train_data)
    exemplo_por_cliente = num_exemplos // args.num_cliente
    tamanho_dataset = [exemplo_por_cliente] * (args.num_cliente - 1)
    tamanho_dataset.append(num_exemplos - sum(tamanho_dataset)) # Garante que todos os dados sejam usados
    
    SEED = 42
    generator = torch.Generator().manual_seed(SEED)
    client_datasets_original = random_split(train_data, tamanho_dataset, generator=generator)
    
    ### MUDANÇA ###
    # 3. Criamos os datasets finais: envenenados para atacantes, normais para os outros
    final_client_datasets = []
    for i in range(args.num_cliente):
        if i < args.num_atacante:
            # Cliente é um atacante: envolvemos seu dataset com PoisonedDataset
            poisoned_set = PoisonedDataset(client_datasets_original[i], trigger_amplitudes, args)
            final_client_datasets.append(poisoned_set)
            print(f"Cliente {i} é um atacante. Dataset envenenado.")
        else:
            # Cliente benigno: usamos o dataset original
            final_client_datasets.append(client_datasets_original[i])

    # 4. Criamos os DataLoaders a partir dos datasets finais
    lista_dataloaders = []
    for dataset in final_client_datasets:
        loader = DataLoader(dataset, args.batchsize, shuffle=True, generator=generator)
        lista_dataloaders.append(loader)

    # Cria o DataLoader de teste
    test_loader = DataLoader(test_data, args.batchsize, shuffle=False)
    
    return test_loader, lista_dataloaders, classes


testSet, trainSetList, classes = pegar_dados_iid()

modeloGlobal = ResNet18().to(device)
listaDeModelos = [ResNet18().to(device) for _ in range(args.num_cliente)]
for modelo in listaDeModelos:
    modelo.load_state_dict(modeloGlobal.state_dict())

listaOptim = [optim.Adam(modelo.parameters(), lr=args.lr) for modelo in listaDeModelos]

# ### MUDANÇA ###
# A chamada para ataqueCifar() é removida daqui, pois a lógica agora está em pegar_dados_iid
# ataqueCifar(trainSetList) 
listaAcuracia = []
errosCertos = []
errosErrados = []
precisaoGatilho = []
try:
    for epoca in range(args.epoca):
        # Seleciona 10 clientes aleatoriamente para treinar
        selecionados = random.sample(range(args.num_cliente), args.selecionar)
        print(f'\n--- Época {epoca} ---')
        print(f"Clientes selecionados: {selecionados}")

        if epoca == 100 or epoca == 200:
            args.lr /= 5
            print(f"Reduzindo taxa de aprendizado para: {args.lr}")
            listaOptim = [optim.Adam(modelo.parameters(), lr=args.lr) for modelo in listaDeModelos]
        
        # Passa o device para a função de treino
        treinar(listaDeModelos, trainSetList, listaOptim, device, selecionados)
        
        print('Agregando modelos...')
        avg(listaDeModelos, modeloGlobal, selecionados)

        print("Testando modelo global...")
        # Passa o device para a função de teste
        acc, backdoor, naoBackdoor, GatilhoEGatilho = testar(modeloGlobal, testSet, device, classes)
        print(f'Acurácia: {acc:.2f}%, ASR (Backdoor): {backdoor:.2f}%, Gatilho C/C: {GatilhoEGatilho:.2f}%, Gatilho C/E: {naoBackdoor:.2f}%')


        listaAcuracia.append(acc)
        errosCertos.append(backdoor)
        errosErrados.append(naoBackdoor)
        precisaoGatilho.append(GatilhoEGatilho)

finally:
    plt.plot(range(len(listaAcuracia)), listaAcuracia)
    plt.title('variacao da acuracia')
    plt.xlabel('epocas')
    plt.ylabel('acuracia')
    plt.savefig('simulacao/variacao_acuracia.png')
    plt.close()

    plt.plot(range(len(errosCertos)), errosCertos)
    plt.title('backdoor atingiu o alvo')
    plt.xlabel('epocas')
    plt.ylabel('erros acertados')
    plt.savefig('simulacao/backdoor_sucesso.png')
    plt.close()

    plt.plot(range(len(errosErrados)), errosErrados)
    plt.title('erros nao atingiu o alvo')
    plt.xlabel('epocas')
    plt.ylabel('erros errados')
    plt.savefig('simulacao/backdoor_fracasso.png')
    plt.close()

    plt.plot(range(len(precisaoGatilho)), precisaoGatilho)
    plt.title('Precisão sobre o gatilho')
    plt.xlabel('Epocas')
    plt.ylabel('Precisão sobre o gatilho')
    plt.savefig('simulacao/precisao_gatilho.png')
    plt.close()



# print("Dados de treino baixados:", train_data)
# print("Dados de teste baixados:", test_data)


# imagem, rotulo = train_data[0]
# imagem_sq = imagem.squeeze()
# img = plt.imshow(imagem_sq, cmap='gray')
# plt.savefig('imagem_retirada.png')

# amp = amplitude(imagem)
