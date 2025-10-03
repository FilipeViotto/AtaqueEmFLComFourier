import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from args import Args
from torch.utils.data import DataLoader, random_split, Subset
import down_datas
args = Args()
from modelos import ResNet18, ModeloCifar10, ModeloCifar10_Revisado
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
from agregacoes import avg, avg_padrao
from treinoTeste import testar, treinar
from enviaEmail import enviarEmail
from ataque import get_trigger_amplitudes, PoisonedDataset
from defesa import defesa
import seaborn as sns
def pegar_dados_iid():
    train_data, test_data, classes = down_datas.down_cifar()  # baixa dados

    trigger_amplitudes = get_trigger_amplitudes(train_data,args)

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

maiores_acc = []
maiores_asr = []
for wm in [10,20,30]:
    historico = {}
    print(f'numero de atacantes {wm}')
    args.num_atacante = wm
    testSet, trainSetList, classes = pegar_dados_iid() # pega dados

    modeloGlobal = ResNet18().to(device)
    listaDeModelos = [ResNet18().to(device) for _ in range(args.num_cliente)]
    for modelo in listaDeModelos:
        modelo.load_state_dict(modeloGlobal.state_dict())

    listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]

    # ### MUDANÇA ###
    # A chamada para ataqueCifar() é removida daqui, pois a lógica agora está em pegar_dados_iid
    # ataqueCifar(trainSetList) 
    listaAcuracia = []
    errosCertos = []
    errosErrados = []
    precisaoGatilho = []
    try:
        for epoca in range(args.epoca):
            selecionados = random.sample(range(args.num_cliente), args.selecionar)
            print(f'\n--- Época {epoca} ---')
            print(f"Clientes selecionados: {selecionados}")

            if epoca == 100 or epoca == 200:
                args.lr = args.reducao
                args.reducao = args.segundaReducao
                print(f"Reduzindo taxa de aprendizado para: {args.lr}")
                listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]
            
            # Passa o device para a função de treino
            treinar(listaDeModelos, trainSetList, listaOptim, device, selecionados, args)

            idModelos, historico = defesa([listaDeModelos[i] for i in selecionados], historico)
            
            print('Agregando modelos...')
            avg(listaDeModelos, modeloGlobal, selecionados, args)


            print("Testando modelo global...")
            # Passa o device para a função de teste
            acc, backdoor, naoBackdoor, matrizDeConfusao, fazermMatriz = testar(modeloGlobal, testSet, device, classes, args=args, fazerMatriz = True if epoca%30==0 else False)
            print(f'Acurácia: {acc:.2f}%, ASR (Backdoor): {backdoor:.2f}%, Gatilho C/E: {naoBackdoor:.2f}%')
            
            if fazermMatriz:
                path = f'simulacoes_res/matrizes/numAtacantes{wm}/'
                os.makedirs(path, exist_ok=True)

                plt.figure(figsize=(10, 8))
                sns.heatmap(matrizDeConfusao, annot=True,fmt='.2f', cmap='Blues', yticklabels= classes, xticklabels=classes)
                plt.tight_layout()
                plt.savefig(f'{path}epoca_{epoca}.png')
                plt.close()

            listaAcuracia.append(acc)
            errosCertos.append(backdoor)
            errosErrados.append(naoBackdoor)


    finally:
        maiores_acc.append(max(listaAcuracia))
        maiores_asr.append(max(errosCertos))
        path = f'simulacoes_res/sim{args.num_atacante}'
        
        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure(figsize=(10, 8))
        config_info = {
            'Itens na Simulação': args.num_cliente,
            'Tipo de modelo': 'ResNet',
            'Taxa de Aprendizagem': '0.1, 0.02, 0.004',
            'Otimizador': 'SDG',
            'Dataset': 'CIFAR-10',
            'Atacantes': wm,
            'Clientes selecionados por epoca': args.selecionar,
            'Peso dos Atacantes': args.fatorLambida
            }
        info_text = '\n'.join([f'{key}: {value}' for key, value in config_info.items()])
        plt.plot(range(len(listaAcuracia)), listaAcuracia, label='Acurácia')
        plt.plot(range(len(errosCertos)), errosCertos, label='ASR')
        plt.plot(range(len(errosErrados)), errosErrados, label='ImprecisãoBackdoor')
        plt.title('Análise de Métricas por Épocas')
        plt.xlabel('Épocas')
        plt.ylabel('Valor da Métrica')
        plt.legend()
        plt.grid(True)
        plt.subplots_adjust(bottom=0.25)
        plt.figtext(0.5, 0.02, info_text, ha="center", fontsize=9, bbox={"facecolor":"lightsteelblue", "alpha":0.5, "pad":5})
        plt.savefig(f'{path}/grafico_combinado.png')
        plt.close()
        #enviarEmail(f'atacante: {wm}\n\nacuracia: {listaAcuracia}\n\nASR: {errosCertos}\n\nimprecisãoBackdoor: {errosErrados}\n', f'{path}/grafico_combinado.png',)


plt.figure(figsize=(10, 8))
config_info = {
    'Itens na Simulação': args.num_cliente,
    'Tipo de modelo': 'ResNet',
    'Taxa de Aprendizagem': '0.1, 0.02, 0.004',
    'Otimizador': 'SDG',
    'Dataset': 'CIFAR-10',
    'Atacantes': wm,
    'Clientes selecionados por epoca': args.selecionar,
    'Peso dos Atacantes': args.fatorLambida
    }
info_text = '\n'.join([f'{key}: {value}' for key, value in config_info.items()])
plt.plot(range(len(maiores_acc)), maiores_acc, label = 'Acurácia')
plt.plot(range(len(maiores_asr)), maiores_asr, label = 'ASR')
plt.title('Análise de Métricas por Atacante')
plt.xlabel('Atacantes')
plt.ylabel('Valor da Métrica')
plt.legend()
plt.grid(True)
plt.subplots_adjust(bottom=0.25)
plt.figtext(0.5, 0.02, info_text, ha="center", fontsize=9, bbox={"facecolor":"lightsteelblue", "alpha":0.5, "pad":5})
plt.savefig(f'metricaPorAtacante.png')
plt.close()
#enviarEmail(f'Resultado total\n\n Maiores acuracias: {maiores_acc}, maiores ASR: {maiores_asr}', f'metricaPorAtacante.png')