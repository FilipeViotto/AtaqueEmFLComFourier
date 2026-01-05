import torch.nn as nn
import torch
from args import Args
from modelos import Modelo
import numpy as np
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score


def treinar(modelos, train_list, listaOptim:list, device, selecionados, args:Args):
    criterion = nn.CrossEntropyLoss()
    for epocaDeTreino in range(args.epocasLocais):
        for i in selecionados:
            modelo = modelos[i]
            conjuntoDeTreino = train_list[i]
            optim = listaOptim[i]
            
            modelo.train()
            for imagem, rotulo in conjuntoDeTreino:
                imagem, rotulo = imagem.to(device), rotulo.to(device)
                
                optim.zero_grad()
                out = modelo(imagem)
                loss = criterion(out, rotulo)
                loss.backward()
                optim.step()
             

def testar(modelo, test_loader, device, classes=None, args = None, fazerMatriz = False):
    matriz_de_confusao = np.zeros((10,10))
    modelo.eval()
    correct = 0
    total = 0
    
    # Métricas de backdoor
    backdoor_success = 0
    backdoor_total = 0
    trigger_correct_as_trigger = 0 # Rotulado como gatilho e era gatilho
    trigger_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = modelo(images)
            _, predicted = torch.max(outputs.data, 1)

            if fazerMatriz:
                for l, p in zip(labels, predicted):
                    matriz_de_confusao[l.item()][p.item()] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #  Análise do Ataque de Backdoor (ASR)
            # Seleciona apenas as imagens que SÃO da classe alvo
            target_mask = (labels == args.alvo)
            if target_mask.sum().item() > 0:
                backdoor_total += target_mask.sum().item()
                # Conta quantas vezes a predição para uma classe alvo foi a classe gatilho
                backdoor_success += (predicted[target_mask] == args.gatilho).sum().item()

            # Análise da performance na classe gatilho original
            # Seleciona apenas as imagens que SÃO da classe gatilho
            trigger_mask = (labels == args.gatilho)
            if trigger_mask.sum().item() > 0:
                trigger_total += trigger_mask.sum().item()
                # Conta quantas vezes o modelo acertou a classe gatilho
                trigger_correct_as_trigger += (predicted[trigger_mask] == args.gatilho).sum().item()

    # Cálculo final das métricas
    accuracy = 100 * correct / total
    asr = 100 * backdoor_success / backdoor_total if backdoor_total > 0 else 0
    
    # Acurácia do modelo em imagens limpas da classe gatilho
    trigger_clean_acc = 100 * trigger_correct_as_trigger / trigger_total if trigger_total > 0 else 0
    # Taxa de erro (classificando como algo diferente do gatilho)
    trigger_error_rate = 100 - trigger_clean_acc

    if fazerMatriz:
        return accuracy, asr, trigger_error_rate, matriz_de_confusao, fazerMatriz
    return accuracy, asr, trigger_error_rate, None, fazerMatriz


def testar2(modelo, test_loader, device = torch.device('cpu'), classes = None):
    modelo.eval()
    
    acertos_totais = 0
    total_amostras = 0
    acertos_por_classe = [0] * len(classes)
    total_por_classe = [0] * len(classes)
    
    with torch.no_grad():
        for imagens, rotulos in test_loader:
            imagens = imagens.to(device)
            rotulos = rotulos.to(device)
            
            saidas = modelo(imagens)
            
            _, predicoes = torch.max(saidas, 1)
            
            total_amostras += rotulos.size(0)
            acertos_totais += (predicoes == rotulos).sum().item()
            
            for i in range(len(rotulos)):
                rotulo_real = rotulos[i]
                predicao = predicoes[i]
                if rotulo_real == predicao:
                    acertos_por_classe[rotulo_real] += 1
                total_por_classe[rotulo_real] += 1

    acuracia_geral = 100 * acertos_totais / total_amostras
    print(f'Acurácia Geral no conjunto de teste: {acuracia_geral:.2f} %')
    print("-" * 30)
    
    for i in range(len(classes)):
        if total_por_classe[i] > 0:
            acuracia_classe = 100 * acertos_por_classe[i] / total_por_classe[i]
            print(f'Acurácia para a classe "{classes[i]}": {acuracia_classe:.2f} %')
            
    return acuracia_geral


def TestarGrupos(listaModelos: list, listaTeste, device):
    votos_finais = {}
    
    if not listaModelos:
        return votos_finais

    for cliente_id, teste_loader in enumerate(listaTeste):
        votos_finais[f'{cliente_id}'] = []
        
        for modelo in listaModelos:
            modelo.eval()
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for images, labels in teste_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = modelo(images)
                    _, predicao = torch.max(outputs.data, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicao.cpu().numpy())

            f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
            
            votos_finais[f'{cliente_id}'].append(f1_micro)

    return votos_finais


def TestarGrupos_F1_Macro(listaModelos: list, listaTeste, device):
    '''o voto é definido pela media dos f1-score de cada classe.'''
    votos_finais = {}

    for cliente_id, teste_loader in enumerate(listaTeste):
        votos_finais[f'{cliente_id}'] = []

        for modelo in listaModelos:
            modelo.eval()
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for images, labels in teste_loader:
                    images, labes = images.to(device), labels.to(device)
                    outputs = modelo(images)
                    _, predicao = torch.max(outputs.data, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicao.cpu().numpy())
            f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            votos_finais[f'{cliente_id}'].append(f1)
    return votos_finais

def TestarGrupos_Minimo(listaModelos: list, listaTeste, device):
    '''Considera o menor valor da classe com menor f1 score como o voto'''

    votos_finais = {}
    
    for cliente_id, teste_loader in enumerate(listaTeste):
        votos_finais[f'{cliente_id}']=[]
        for modelo in listaModelos:
            modelo.eval()
            all_labels = []
            all_predictions = []

            with torch.no_grad():
                for images, labels in teste_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = modelo(images)
                    _, predicao = torch.max(outputs.data, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicao.cpu().numpy())

            f1_por_classe = f1_score(all_labels, all_predictions, average=None, zero_division=0)

            pior_classe_score = np.min(f1_por_classe)
            votos_finais[f'{cliente_id}'].append(float(pior_classe_score))
    return votos_finais


def ColetarMetricas_Detalhadas(listaModelos: list, listaTeste, device):
    """
    Retorna um dicionário detalhado para análise.
    Estrutura:
    dados[cliente_id][modelo_grupo_id] = {
       'macro': float,
       'per_class': [f1_c0, f1_c1, ...],
       'accuracy': float
    }
    """
    dados_analise = {}

    for cliente_id, teste_loader in enumerate(listaTeste):
        dados_analise[f'{cliente_id}'] = []

        for modelo in listaModelos:
            modelo.eval()
            all_labels = []
            all_predictions = []
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in teste_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = modelo(images)
                    _, predicao = torch.max(outputs.data, 1)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicao.cpu().numpy())
                    
                    total += labels.size(0)
                    correct += (predicao == labels).sum().item()

            # F1 Score por classe (retorna um array de 10 posições para CIFAR-10)
            f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
            f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            acc = correct / total if total > 0 else 0

            metricas_modelo = {
                'macro': f1_macro,
                'per_class': f1_per_class.tolist(), # Converte para lista Python pura
                'accuracy': acc
            }
            dados_analise[f'{cliente_id}'].append(metricas_modelo)
            
    return dados_analise