import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import os
import os.path
from collections import OrderedDict
import copy

# Importações
from args import Args
from torch.utils.data import DataLoader, random_split
import down_datas
from modelos import ModeloCifar10_Revisado
from agregacoes import avg
from treinoTeste import testar, treinar, TestarGrupos_F1_Macro
from ataque import get_trigger_amplitudes, PoisonedDataset
from defesa2 import agregaGrupos

# Configurações Iniciais
args = Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Garante que a taxa de aprendizado do servidor exista (padrão 1.0)
if not hasattr(args, 'taxaDeAprendizadoDoServidor'):
    args.taxaDeAprendizadoDoServidor = 1.0

def pegar_dados_iid():
    """
    Prepara dados CIFAR-10 IID com envenenamento (Backdoor) para atacantes.
    """
    train_data, test_data, classes = down_datas.down_cifar()
    trigger_amplitudes = get_trigger_amplitudes(train_data, args)

    # Divisão do Treino
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
            # Cliente Atacante: Dados Envenenados
            poisoned_set = PoisonedDataset(client_datasets_original[i], trigger_amplitudes, args)
            final_client_datasets_treino.append(poisoned_set)
        else:
            # Cliente Honesto: Dados Normais
            final_client_datasets_treino.append(client_datasets_original[i])

    lista_dataloaders_treino = [DataLoader(ds, args.batchsize, shuffle=True, generator=generator) for ds in final_client_datasets_treino]

    # Dataset Global de Teste (para medir ASR real)
    dataset_teste_completo = DataLoader(test_data, args.batchsize, shuffle=False)
    
    # Dataset de Teste particionado (para validação local dos clientes)
    num_exemplos_teste = len(test_data)
    exemplo_por_cliente_teste = num_exemplos_teste // args.num_cliente
    tamanho_dataset_teste = [exemplo_por_cliente_teste] * (args.num_cliente - 1)
    tamanho_dataset_teste.append(num_exemplos_teste - sum(tamanho_dataset_teste))
    
    dataset_splits_teste = random_split(test_data, tamanho_dataset_teste, generator=generator)
    lista_dataloaders_teste = [DataLoader(ds, args.batchsize, shuffle=False) for ds in dataset_splits_teste]

    return dataset_teste_completo, lista_dataloaders_teste, lista_dataloaders_treino, classes

# --- Loop Principal ---

pasta_graficos_main = "Graficos_Avaliacao_Defesa"
os.makedirs(pasta_graficos_main, exist_ok=True)

# Loop comparativo: COM defesa vs SEM defesa
for args.comDefesa in [True, False]:
    
    cenario = 'ComDefesa' if args.comDefesa else 'SemDefesa'


    
    for wm in [0, 4, 10]: 
        
        args.num_atacante = wm
        pasta_execucao_atual = os.path.join(pasta_graficos_main, f"{cenario}_atacantes_{wm}")
        os.makedirs(pasta_execucao_atual, exist_ok=True)
        
        testSetGlobal, testSetClientes, trainSetList, classes = pegar_dados_iid() 

        modeloGlobal = ModeloCifar10_Revisado().to(device)
        listaDeModelos = [ModeloCifar10_Revisado().to(device) for _ in range(args.num_cliente)]
        
        for modelo in listaDeModelos:
            modelo.load_state_dict(modeloGlobal.state_dict())

        listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]

        # Lista fixa de seleção para garantir consistência na comparação
        selecionados = [10,0,1,2,3,11,14,6,4,7,12,15,17,5,8,13,16,18,19,9]
        # Se quiser usar todos, descomente: selecionados = list(range(args.num_cliente))

        historico_acc = []
        historico_asr = []

        for epoca in range(args.epoca):
            
            # Redução LR
            if epoca == 100 or epoca == 200:
                args.lr = args.reducao
                listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]
            
            # 1. Treinamento Local
            treinar(listaDeModelos, trainSetList, listaOptim, device, selecionados, args)

            # 2. Topologia de Grupos
            gruposDeModelos = [[] for _ in range(args.numGrupos)]
            contagemAtacantes = [0] * args.numGrupos
            
            for i, clienteId in enumerate(selecionados):
                grupoId = i % args.numGrupos
                gruposDeModelos[grupoId].append(listaDeModelos[clienteId])
                if clienteId < args.num_atacante:
                    contagemAtacantes[grupoId] += 1
            
            menorQtdAtacante = min(contagemAtacantes)

            modelosAgregados = agregaGrupos(gruposDeModelos, modeloGlobal)

            votos = TestarGrupos_F1_Macro(modelosAgregados, testSetClientes, device)
            
            vezesQueEscolheuMenosAtacantes = 0
            
            if args.comDefesa:
                
                for i in range(args.num_cliente):
                    istr = str(i)
                    if istr in votos and votos[istr]:
                        votosAtual = votos[istr]
                        indiceVencedor = votosAtual.index(max(votosAtual))
                        
                        # Monitoramento da Defesa
                        if contagemAtacantes[indiceVencedor] == menorQtdAtacante:
                            vezesQueEscolheuMenosAtacantes += 1

                        # Atualiza local com o Vencedor (descarta os ruins)
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

            # Print formatado para facilitar leitura
            def_status = "[DEFESA ON]" if args.comDefesa else "[DEFESA OFF]"
            print(f"Epoca {epoca:03d} {def_status} | Acc: {acc_global:6.2f}% | ASR: {asr_global:6.2f}% | Escolha Segura: {vezesQueEscolheuMenosAtacantes}/{args.num_cliente}")

        # --- Geração dos Gráficos Finais ---
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
        
        # Log em TXT
        with open(os.path.join(pasta_execucao_atual, "metricas.txt"), "w") as f:
            f.write("Epoca,ACC_Clean,ASR_Attack\n")
            for e, (acc, asr) in enumerate(zip(historico_acc, historico_asr)):
                f.write(f"{e},{acc},{asr}\n")

print("\nExecução finalizada. Compare os gráficos 'ComDefesa' vs 'SemDefesa'.")