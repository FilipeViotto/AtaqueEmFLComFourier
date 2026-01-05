import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from args import Args
from torch.utils.data import DataLoader, random_split, TensorDataset
import down_datas
from modelos import ModeloCifar10_Revisado
import torch.optim as optim
import os
import math
from scipy import stats

# Assumindo que esses arquivos existem e estão corretos
from defesa2 import agregaGrupos
from treinoTeste import TestarGrupos, treinar, TestarGrupos_F1_Macro, TestarGrupos_Minimo
from ataque import get_trigger_amplitudes, PoisonedDataset, aplicar_trigger_a_dataset

# --- Configuração Inicial ---
args = Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def pegar_dados_iid(num_atacantes):
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
        if i < num_atacantes:
            poisoned_set = PoisonedDataset(client_datasets_original[i], trigger_amplitudes, args)
            final_client_datasets_treino.append(poisoned_set)
        else:
            final_client_datasets_treino.append(client_datasets_original[i])
    lista_dataloaders_treino = [DataLoader(ds, args.batchsize, shuffle=True, generator=generator) for ds in final_client_datasets_treino]

    # Divisão do Conjunto de Teste (local para cada cliente)
    num_exemplos_teste = len(test_data)
    exemplo_por_cliente_teste = num_exemplos_teste // args.num_cliente
    tamanho_dataset_teste = [exemplo_por_cliente_teste] * (args.num_cliente - 1)
    tamanho_dataset_teste.append(num_exemplos_teste - sum(tamanho_dataset_teste))
    dataset_splits_teste = random_split(test_data, tamanho_dataset_teste, generator=generator)
    lista_dataloaders_teste_local = [DataLoader(ds, args.batchsize, shuffle=False) for ds in dataset_splits_teste]

    # Cria um dataloader de teste global para a avaliação final
    test_loader_global = DataLoader(test_data, batch_size=args.batchsize, shuffle=False)
    
    # Cria um dataset de teste para ASR
    test_data_backdoor = aplicar_trigger_a_dataset(test_data, trigger_amplitudes, args)
    test_loader_backdoor = DataLoader(test_data_backdoor, batch_size=args.batchsize, shuffle=False)


    return lista_dataloaders_teste_local, lista_dataloaders_treino, classes, test_loader_global, test_loader_backdoor

def avaliar_sistema_por_voto(lista_de_modelos, test_loader, device):

    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Recolhe a previsão de cada modelo para o lote de imagens
        predicoes_modelos = []
        for modelo in lista_de_modelos:
            modelo.eval()
            with torch.no_grad():
                outputs = modelo(images)
                _, predicted = torch.max(outputs.data, 1)
                predicoes_modelos.append(predicted.cpu().numpy())
        
        # Transpõe para ter (num_imagens, num_modelos)
        predicoes_modelos = np.array(predicoes_modelos).T
        
        # Realiza o voto majoritário para cada imagem
        # stats.mode retorna o valor mais frequente e a sua contagem
        voto_final, _ = stats.mode(predicoes_modelos, axis=1, keepdims=False)
        
        voto_final_tensor = torch.tensor(voto_final).to(device)
        
        total += labels.size(0)
        correct += (voto_final_tensor == labels).sum().item()
        
    return 100 * correct / total if total > 0 else 0


# --- Armazenamento de Resultados para Análise Final ---
resultados_finais = {}

# --- Loop Principal de Simulação ---
cenarios = {'ComDefesa': True, 'SemDefesa': False}

for cenario, com_defesa in cenarios.items():
    print(f"\nEXECUTANDO CENÁRIO: {cenario}")
    print("=========================================")
    
    cenario_accuracies = []
    cenario_asrs = []
    lista_atacantes = [0, 10, 20, 30, 40, 50]

    for wm in lista_atacantes: 
        print(f'\n--- Simulação para {wm} atacantes ---')
        args.num_atacante = wm
        
        testSetLocal, trainSetList, classes, testSetGlobal, testSetBackdoor = pegar_dados_iid(wm)

        modeloGlobalTemplate = ModeloCifar10_Revisado().to(device)
        listaDeModelos = [ModeloCifar10_Revisado().to(device) for _ in range(args.num_cliente)]
        
        for modelo in listaDeModelos:
            modelo.load_state_dict(modeloGlobalTemplate.state_dict())

        listaOptim = [optim.SGD(modelo.parameters(), lr=args.lr, momentum=0.9) for modelo in listaDeModelos]

        for epoca in range(args.epoca):
            if epoca % 20 == 0:
                print(f'  Época {epoca}/{args.epoca - 1}')

            selecionados = random.sample(range(args.num_cliente), args.selecionar)
            treinar(listaDeModelos, trainSetList, listaOptim, device, selecionados, args)

            if com_defesa:
                gruposDeModelos = [[] for _ in range(args.numGrupos)]
                for i, clienteId in enumerate(selecionados):
                    gruposDeModelos[i % args.numGrupos].append(listaDeModelos[clienteId])
                
                modelosAgregados = agregaGrupos(gruposDeModelos, modeloGlobalTemplate)
                if not modelosAgregados: continue

                votos = TestarGrupos_F1_Macro(modelosAgregados, testSetLocal, device)
                
                for i in range(args.num_cliente):
                    istr = str(i)
                    if istr in votos and votos[istr]:
                        votosAtual = votos[istr]
                        indiceDoModeloEscolhido = votosAtual.index(max(votosAtual))
                        modeloVencedor = modelosAgregados[indiceDoModeloEscolhido]
                        listaDeModelos[i].load_state_dict(modeloVencedor.state_dict())
            else:
                # Lógica sem defesa: agregação FedAvg simples
                avg_dict = modeloGlobalTemplate.state_dict()
                for key in avg_dict:
                    if avg_dict[key].is_floating_point():
                        avg_dict[key] = torch.mean(torch.stack([listaDeModelos[i].state_dict()[key] for i in selecionados]), dim=0)
                
                # Distribui o modelo agregado para todos
                for modelo in listaDeModelos:
                    modelo.load_state_dict(avg_dict)

        # --- Avaliação Final após todas as épocas ---
        print("  Avaliando desempenho final do sistema...")
        final_acc = avaliar_sistema_por_voto(listaDeModelos, testSetGlobal, device)
        final_asr = avaliar_sistema_por_voto(listaDeModelos, testSetBackdoor, device)
        
        print(f'  -> Acurácia Final: {final_acc:.2f}%')
        print(f'  -> ASR Final: {final_asr:.2f}%')
        
        cenario_accuracies.append(final_acc)
        cenario_asrs.append(final_asr)

    resultados_finais[cenario] = {
        'atacantes': lista_atacantes,
        'acuracia': cenario_accuracies,
        'asr': cenario_asrs
    }

# --- Plotagem do Gráfico Comparativo Final ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

cores = {'ComDefesa': ('#28a745', '#17a2b8'), 'SemDefesa': ('#dc3545', '#ffc107')}
estilos = {'Acurácia': '-', 'ASR': '--'}

for cenario, data in resultados_finais.items():
    ax.plot(data['atacantes'], data['acuracia'], marker='o', linestyle=estilos['Acurácia'], color=cores[cenario][0], label=f'Acurácia ({cenario})')
    ax.plot(data['atacantes'], data['asr'], marker='x', linestyle=estilos['ASR'], color=cores[cenario][1], label=f'ASR ({cenario})')

ax.set_title('Eficácia da Defesa vs. Percentagem de Atacantes', fontsize=18, pad=20)
ax.set_xlabel('Número de Clientes Atacantes', fontsize=14)
ax.set_ylabel('Percentagem (%)', fontsize=14)
ax.set_xticks(lista_atacantes)
ax.set_ylim(0, 101)
ax.legend(fontsize=12, loc='best', frameon=True, shadow=True)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("analise_final_defesa.png", dpi=300)
plt.show()