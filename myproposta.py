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
from agregacoes import avg, avg_padrao, avgComDefesa
from treinoTeste import testar, treinar, TestarGrupos
from enviaEmail import enviarEmail
from ataque import get_trigger_amplitudes, PoisonedDataset
from defesa import defesa
import seaborn as sns
from collections import OrderedDict
from defesa2 import agregaGrupos, selecionarNovoModeloGlobal
import matplotlib.patches as mpatches

def pegar_dados_iid():
    """
    Prepara e distribui os datasets CIFAR-10 para os clientes de forma IID.
    Aplica envenenamento de dados para clientes atacantes.
    """
    train_data, test_data, classes = down_datas.down_cifar()
    trigger_amplitudes = get_trigger_amplitudes(train_data, args)

    # Divisão do Conjunto de Treino
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

    # Divisão do Conjunto de Teste
    num_exemplos_teste = len(test_data)
    exemplo_por_cliente_teste = num_exemplos_teste // args.num_cliente
    tamanho_dataset_teste = [exemplo_por_cliente_teste] * (args.num_cliente - 1)
    tamanho_dataset_teste.append(num_exemplos_teste - sum(tamanho_dataset_teste))
    
    if any(t == 0 for t in tamanho_dataset_teste):
        raise ValueError("Erro na divisão de dados de teste resultou em um subconjunto de tamanho 0.")

    dataset_splits_teste = random_split(test_data, tamanho_dataset_teste, generator=generator)
    lista_dataloaders_teste = [DataLoader(ds, args.batchsize, shuffle=False) for ds in dataset_splits_teste]

    return lista_dataloaders_teste, lista_dataloaders_treino, classes

pasta_graficos_main = "graficos_resultados"
os.makedirs(pasta_graficos_main, exist_ok=True)
for args.comDefesa in [True]:
    maiores_acc = []
    maiores_asr = []
    acertosDefesa = []
    
    cenario = 'ComDefesa' if args.comDefesa else 'SemDefesa'
    
    for wm in [10,20,30,40,50]: 
        print(f'numero de atacantes {wm}')
        args.num_atacante = wm
        
        pasta_execucao_atual = os.path.join(pasta_graficos_main, f"{cenario}_atacantes_{wm}")
        
        os.makedirs(pasta_execucao_atual, exist_ok=True)
        
        testSet, trainSetList, classes = pegar_dados_iid() # pega dados

        modeloGlobal = ModeloCifar10_Revisado().to(device)
        listaDeModelos = [ModeloCifar10_Revisado().to(device) for _ in range(args.num_cliente)]
        
        for modelo in listaDeModelos:
            modelo.load_state_dict(modeloGlobal.state_dict())

        listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]

        selecionados = random.sample(range(args.num_cliente), args.selecionar)
        selecionados = list(selecionados)
        random.shuffle(selecionados)

        for epoca in range(args.epoca):
            
            print(f'\n--- Época {epoca} ---')

            if epoca == 100 or epoca == 200:
                args.lr = args.reducao
                args.reducao = args.segundaReducao
                print(f"Reduzindo taxa de aprendizado para: {args.lr}")
                listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]
            
            treinar(listaDeModelos, trainSetList, listaOptim, device, selecionados, args)

            gruposDeModelos = [[] for _ in range(args.numGrupos)]
            
            contagemAtacantes = [0]*args.numGrupos
            
            import math
            tamanhoDoGrupo = math.ceil(len(selecionados)/args.numGrupos)
            clientesPorGrupo = [[] for _ in range(args.numGrupos)]
            
            for i, clienteId in enumerate(selecionados):
                grupoId = i%args.numGrupos
                gruposDeModelos[grupoId].append(listaDeModelos[clienteId])
                # clientesPorGrupo[grupoId].append(clienteId)
                if clienteId<args.num_atacante:
                    contagemAtacantes[grupoId] += 1
            

                # inicio = i*tamanhoDoGrupo
                # fim = inicio + tamanhoDoGrupo
                # clientes = selecionados[inicio:fim]
                # quantidadeEmCadaGrupo.append(len(clientes))
                # for selecionado in clientes:
                #     gruposDeModelos[i].append(listaDeModelos[selecionado])
                #     if selecionado < args.num_atacante:
                #         contagemAtacantes[i] += 1
            


            modelosAgregados = agregaGrupos(gruposDeModelos, modeloGlobal)

            print("Testando modelo global...")
            
            votos = TestarGrupos(modelosAgregados, testSet, device)
            
            # novoModeloGlobalDict, indiceVencedor, somaPontuacao = selecionarNovoModeloGlobal(votos, modelosAgregados, args)

            eleicaoTotal = [0] * len(modelosAgregados)

            for i in range(args.num_cliente):
                istr = str(i)
                if istr in votos and votos[istr]:
                    votosAtual = votos[istr]
                    indiceDoModeloEscolhido = votosAtual.index(max(votosAtual))

                    modeloVencedor = modelosAgregados[indiceDoModeloEscolhido]
                    listaDeModelos[i].load_state_dict(modeloVencedor.state_dict())

                    eleicaoTotal[indiceDoModeloEscolhido] += 1
                    

            # modeloGlobal.load_state_dict(novoModeloGlobalDict)



            # for modelo in listaDeModelos:
            #     modelo.load_state_dict(modeloGlobal.state_dict())
            
            if epoca%5==0:
                for cliente, voto in votos.items():
                    
                    maiorScore = max(voto)
                    indicelocalVencedor = voto.index(maiorScore)
                    if epoca%1 == 0:
                        # f1_score = [v['weighted'] for v in votos]
                        gruposIndices = np.arange(len(voto))

                        grupoComprometido = [False] * args.numGrupos
                        for i in range(args.num_cliente):
                            if i < args.num_atacante:
                                # Usando uma variável diferente ('indice_do_grupo') para não sobrescrever 'gruposIndices'
                                indice_do_grupo = i % args.numGrupos
                                grupoComprometido[indice_do_grupo] = True

                        colors = ['#d9534f' if contagemAtacantes[i]>0 else '#5bc0de' for i in gruposIndices]

                        plt.figure(figsize=(16,7))
                        bars = plt.bar(gruposIndices, voto, color=colors, label='F1-Score do Grupo')

                        for i, bar in enumerate(bars):
                            count = contagemAtacantes[i]
                            if count > 0:
                                yval = bar.get_height()
                                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{count}A', 
                                        ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

                        if args.comDefesa and indicelocalVencedor !=-1:
                            bars[indicelocalVencedor].set_edgecolor('yellow')
                            bars[indicelocalVencedor].set_linewidth(3)

                        plt.xlabel('indice do grupo agregado', fontsize=10)
                        plt.ylabel('F1-Score', fontsize=10)
                        plt.title(f"Desempenho dos Grupos na Época {epoca}\n({cenario}, {wm} atacantes)", fontsize=16)
                        plt.xticks(gruposIndices)
                        plt.ylim(0,1.05)
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        

                        red_patch = mpatches.Patch(color='#d9534f', label='Grupo Comprometido')
                        blue_patch = mpatches.Patch(color='#5bc0de', label='Grupo Benigno')
                        yellow_border_patch = mpatches.Patch(edgecolor='yellow', facecolor='none', linewidth=3, label='Modelo Escolhido')

                        handles = [red_patch, blue_patch]
                        if args.comDefesa and indicelocalVencedor != -1:
                            handles.append(yellow_border_patch)
                        plt.legend(handles=handles)

                        plt.tight_layout()
                        
                        plot_path = os.path.join(pasta_execucao_atual, f"epoca_{epoca}")

                        os.makedirs(plot_path, exist_ok=True)

                        pasta = os.path.join(plot_path, f"cliente_{cliente}.png")
                        
                        plt.savefig(pasta)
                        plt.close()

            # indicelocalVencedor = indiceVencedor
            # if epoca%1 == 0:
            #     # f1_score = [v['weighted'] for v in votos]
            #     gruposIndices = np.arange(len(somaPontuacao))

            #     grupoComprometido = [False] * args.numGrupos
            #     for i in range(args.num_cliente):
            #         if i < args.num_atacante:
            #             # Usando uma variável diferente ('indice_do_grupo') para não sobrescrever 'gruposIndices'
            #             indice_do_grupo = i % args.numGrupos
            #             grupoComprometido[indice_do_grupo] = True

            #     colors = ['#d9534f' if contagemAtacantes[i]>0 else '#5bc0de' for i in gruposIndices]

            #     plt.figure(figsize=(16,7))
            #     bars = plt.bar(gruposIndices, somaPontuacao, color=colors, label='F1-Score do Grupo')

            #     for i, bar in enumerate(bars):
            #         count = contagemAtacantes[i]
            #         if count > 0:
            #             yval = bar.get_height()
            #             plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{count}A', 
            #                     ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')

            #     if args.comDefesa and indicelocalVencedor !=-1:
            #         bars[indicelocalVencedor].set_edgecolor('yellow')
            #         bars[indicelocalVencedor].set_linewidth(3)

            #     plt.xlabel('indice do grupo agregado', fontsize=10)
            #     plt.ylabel('F1-Score', fontsize=10)
            #     plt.title(f"Desempenho dos Grupos na Época {epoca}\n({cenario}, {wm} atacantes)", fontsize=16)
            #     plt.xticks(gruposIndices)
            #     plt.ylim(0,1.05)
            #     plt.grid(axis='y', linestyle='--', alpha=0.7)
                

            #     red_patch = mpatches.Patch(color='#d9534f', label='Grupo Comprometido')
            #     blue_patch = mpatches.Patch(color='#5bc0de', label='Grupo Benigno')
            #     yellow_border_patch = mpatches.Patch(edgecolor='yellow', facecolor='none', linewidth=3, label='Modelo Escolhido')

            #     handles = [red_patch, blue_patch]
            #     if args.comDefesa and indicelocalVencedor != -1:
            #         handles.append(yellow_border_patch)
            #     plt.legend(handles=handles)

            #     plt.tight_layout()
            #     plot_path = os.path.join(pasta_execucao_atual, f"epoca_{epoca}.png")
            #     plt.savefig(plot_path)
            #     plt.close()



