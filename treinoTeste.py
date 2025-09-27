import torch.nn as nn
import torch
from args import Args
from modelos import Modelo


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
        
                    

def testar(modelo, test_loader, device, classes=None, args = None):
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
            
            # 1. Acurácia Geral
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 2. Análise do Ataque de Backdoor (ASR)
            # Seleciona apenas as imagens que SÃO da classe alvo
            target_mask = (labels == args.alvo)
            if target_mask.sum().item() > 0:
                backdoor_total += target_mask.sum().item()
                # Conta quantas vezes a predição para uma classe alvo foi a classe gatilho
                backdoor_success += (predicted[target_mask] == args.gatilho).sum().item()

            # 3. Análise da performance na classe gatilho original
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
    
    # O retorno corresponde à sua chamada no main.py:
    # acc, backdoor, naoBackdoor, GatilhoEGatilho
    return accuracy, asr, trigger_error_rate


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
        else:
            print(f'Nenhuma amostra da classe "{classes[i]}" foi encontrada no teste.')
            
    return acuracia_geral