import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import os
import os.path
from collections import OrderedDict
import copy

from args import Args
from torch.utils.data import DataLoader, random_split
import down_datas
from modelos import ModeloCifar10_Revisado
from agregacoes import avg
from treinoTeste import testar, treinar, TestarGrupos
from ataque import get_trigger_amplitudes, PoisonedDataset
from defesa2 import agregaGrupos

args = Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not hasattr(args, 'taxaDeAprendizadoDoServidor'):
    args.taxaDeAprendizadoDoServidor = 1.0

def pegar_dados_iid():
    train_data, test_data, classes = down_datas.down_cifar()
    trigger_amplitudes = get_trigger_amplitudes(train_data, args)

    num_exemplos_treino = len(train_data)
    exemplo_por_cliente_treino = num_exemplos_treino // args.num_cliente
    tamanho_dataset_treino = [exemplo_por_cliente_treino] * (args.num_cliente - 1)
    tamanho_dataset_treino.append(num_exemplos_treino - sum(tamanho_dataset_treino))

    SEED = 42
    generator = torch.Generator().manual_seed(SEED)
    client_datasets_original = random_split(train_data, tamanho_dataset_treino, generator=generator)

    final_client_datasets_treino = []
    for i in range(args.num_cliente):
        if i < args.num_atacante:
            poisoned_set = PoisonedDataset(client_datasets_original[i], trigger_amplitudes, args)
            final_client_datasets_treino.append(poisoned_set)
        else:
            final_client_datasets_treino.append(client_datasets_original[i])

    lista_dataloaders_treino = [DataLoader(ds, args.batchsize, shuffle=True, generator=generator) for ds in final_client_datasets_treino]
    dataset_teste_completo = DataLoader(test_data, args.batchsize, shuffle=False)
    
    num_exemplos_teste = len(test_data)
    exemplo_por_cliente_teste = num_exemplos_teste // args.num_cliente
    tamanho_dataset_teste = [exemplo_por_cliente_teste] * (args.num_cliente - 1)
    tamanho_dataset_teste.append(num_exemplos_teste - sum(tamanho_dataset_teste))
    
    dataset_splits_teste = random_split(test_data, tamanho_dataset_teste, generator=generator)
    lista_dataloaders_teste = [DataLoader(ds, args.batchsize, shuffle=False) for ds in dataset_splits_teste]

    return dataset_teste_completo, lista_dataloaders_teste, lista_dataloaders_treino, classes

pasta_graficos_main = "Graficos_Avaliacao_Defesa_ComDefesaBlenda"
os.makedirs(pasta_graficos_main, exist_ok=True)
for args.comDefesa in [True]:
    
    cenario = 'ComDefesa' if args.comDefesa else 'SemDefesa'
    
    for wm in [10]: 
        
        args.num_atacante = wm
        pasta_execucao_atual = os.path.join(pasta_graficos_main, f"{cenario}_atacantes_{wm}")
        os.makedirs(pasta_execucao_atual, exist_ok=True)
        
        testSetGlobal, testSetClientes, trainSetList, classes = pegar_dados_iid()

        modeloGlobal = ModeloCifar10_Revisado().to(device)
        listaDeModelos = [ModeloCifar10_Revisado().to(device) for _ in range(args.num_cliente)]
        
        for modelo in listaDeModelos:
            modelo.load_state_dict(modeloGlobal.state_dict())

        listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]

        selecionados = [10,0,1,2,3,11,14,6,4,7,12,15,17,5,8,13,16,18,19,9]
        # selecionados = list(range(args.num_cliente))

        historico_acc = []
        historico_asr = []

        for epoca in range(args.epoca):
            
            if epoca == 100 or epoca == 200:
                args.lr = args.reducao
                listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]
            
            treinar(listaDeModelos, trainSetList, listaOptim, device, selecionados, args)

            gruposDeModelos = [[] for _ in range(args.numGrupos)]
            contagemAtacantes = [0] * args.numGrupos
            
            for i, clienteId in enumerate(selecionados):
                grupoId = i % args.numGrupos
                gruposDeModelos[grupoId].append(listaDeModelos[clienteId])
                if clienteId < args.num_atacante:
                    contagemAtacantes[grupoId] += 1
            
            menorQtdAtacante = min(contagemAtacantes)

            modelosAgregados = agregaGrupos(gruposDeModelos, modeloGlobal)

            votos = TestarGrupos(modelosAgregados, testSetClientes, device)
            
            vezesQueEscolheuMenosAtacantes = 0
            
            if args.comDefesa:
                for i in range(args.num_cliente):
                    istr = str(i)
                    if istr in votos and votos[istr]:
                        votosAtual = votos[istr]
                        indiceVencedor = votosAtual.index(max(votosAtual))
                        
                        if contagemAtacantes[indiceVencedor] == menorQtdAtacante:
                            vezesQueEscolheuMenosAtacantes += 1

                        listaDeModelos[i].load_state_dict(modelosAgregados[indiceVencedor].state_dict())
                
                avg(listaDeModelos, modeloGlobal, selecionados, args)

            else:
                avg(listaDeModelos, modeloGlobal, selecionados, args)
                vezesQueEscolheuMenosAtacantes = 0
            
            acc_global, asr_global, _, _, _ = testar(
                modeloGlobal, testSetGlobal, device, classes, args, fazerMatriz=False
            )
            
            historico_acc.append(acc_global)
            historico_asr.append(asr_global)

            def_status = "[DEFESA ON]" if args.comDefesa else "[DEFESA OFF]"
            print(f"Epoca {epoca:03d} {def_status} | Acc: {acc_global:6.2f}% | ASR: {asr_global:6.2f}% | Escolha Segura: {vezesQueEscolheuMenosAtacantes}/{args.num_cliente}")

            if epoca % 5 == 0:
                contagem_votos_grupos = [0] * args.numGrupos
                
                for cliente_id_str, lista_scores in votos.items():
                    if lista_scores:
                        
                        indice_escolhido = lista_scores.index(max(lista_scores))
                        
                        contagem_votos_grupos[indice_escolhido] += 1
                
                gruposIndices = np.arange(args.numGrupos)
                
                colors = ['#d9534f' if contagemAtacantes[i] > 0 else '#5bc0de' for i in gruposIndices]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(gruposIndices, contagem_votos_grupos, color=colors, edgecolor='black', alpha=0.8)
                
                for idx, rect in enumerate(bars):
                    height = rect.get_height()

                    if height > 0:
                        plt.text(rect.get_x() + rect.get_width()/2., height + 0.2,
                                f'{int(height)} Votos',
                                ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
                    qtd_atk = contagemAtacantes[idx]
                    label_atk = f"{qtd_atk} Atacantes" if qtd_atk > 0 else "Limpo"
                    cor_texto = 'white' if (height > 2 and qtd_atk > 0) else 'black'
                    pos_y = height/2 if height > 2 else -1
                    
                    if height > 0:
                        plt.text(rect.get_x() + rect.get_width()/2., height/2,
                                f'{qtd_atk} Atk',
                                ha='center', va='center', color='white', fontweight='bold')
                
                plt.title(f"Distribuição de Escolhas dos Clientes - Época {epoca}\nCenário: {cenario} ({wm} Atacantes)")
                plt.xlabel("ID do Grupo (Candidato)")
                plt.ylabel("Quantidade de Clientes que Escolheram este Grupo")
                plt.xticks(gruposIndices)
                plt.ylim(0, args.num_cliente + 2)
                plt.grid(axis='y', linestyle='--', alpha=0.5)
                
                red_patch = plt.Rectangle((0,0),1,1, color='#d9534f', label='Grupo com Atacantes')
                blue_patch = plt.Rectangle((0,0),1,1, color='#5bc0de', label='Grupo Limpo')
                plt.legend(handles=[red_patch, blue_patch])
                
                caminho_plot = os.path.join(pasta_execucao_atual, f"votos_epoca_{epoca}.png")
                plt.savefig(caminho_plot)
                plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(historico_acc, label='Acurácia (Tarefa Principal)', linewidth=2)
        plt.plot(historico_asr, label='ASR (Sucesso do Ataque)', color='red', linestyle='--', linewidth=2)
        
        plt.xlabel('Épocas')
        plt.ylabel('Porcentagem (%)')
        plt.title(f'Performance: {cenario} ({wm} Atacantes)\nData Poisoning Puro')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 105)
        
        nome_arquivo = f"resultado_{cenario}_{wm}_atk.png"
        plt.savefig(os.path.join(pasta_execucao_atual, nome_arquivo))
        plt.close()
        
        with open(os.path.join(pasta_execucao_atual, "metricas.txt"), "w") as f:
            f.write("Epoca,ACC_Clean,ASR_Attack\n")
            for e, (acc, asr) in enumerate(zip(historico_acc, historico_asr)):
                f.write(f"{e},{acc},{asr}\n")

