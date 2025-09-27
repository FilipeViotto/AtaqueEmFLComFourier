import torch.nn as nn
import torch
from args import Args
args = Args()
from modelos import Modelo


def treinar(modelos, train_list, listaOptim:list, device, selecionados):
    criterion = nn.CrossEntropyLoss()
    for epocaDeTreino in range(args.epocasLocais):
        print(f"epoca interna: {epocaDeTreino}")
        for i in selecionados:
            modelo = modelos[i]
            conjuntoDeTreino = train_list[i]
            optim = listaOptim[i]
            
            modelo.train()
            for imagem, rotulo in conjuntoDeTreino:
                # Agora 'imagem' e 'rotulo' são garantidamente tensores!
                imagem, rotulo = imagem.to(device), rotulo.to(device)
                
                optim.zero_grad()
                out = modelo(imagem)
                loss = criterion(out, rotulo)
                loss.backward()
                optim.step()

def testar(modelo, teste, device = torch.device('cpu'), classes=None):
    modelo.eval()
    totalUm = 0
    totalGatilho = 0
    erroPorBack = 0
    outrosErros = 0
    acertos_totais = 0
    total_amostras = 0
    disseSerGatilho = 0

    with torch.no_grad():
        for x, y in teste:
            x = x.to(device)
            y = y.to(device)

            out = modelo(x)
            _, predictions = out.max(1)

            for i in range(len(y)):
                true_label = y[i]
                predLabel = predictions[i]

                if true_label == predLabel:
                    acertos_totais += 1

                if true_label == args.alvo:
                    totalUm += 1
                    if predLabel == args.gatilho:
                        erroPorBack += 1
                    elif predLabel != args.alvo:
                        outrosErros += 1
                
                elif true_label == args.gatilho:
                    totalGatilho += 1
                
                if predLabel == args.gatilho:
                    disseSerGatilho += 1
            
            total_amostras += y.size(0)

    # Evitar divisão por zero se não houver amostras
    if totalUm == 0: totalUm = 1
    if disseSerGatilho == 0: disseSerGatilho = 1
    if total_amostras == 0: total_amostras = 1

    # Cálculos de acurácia corrigidos
    acuracia = (acertos_totais / total_amostras) * 100
    backdoor_rate = (erroPorBack / totalUm) * 100
    outros_erros_rate = (outrosErros / totalUm) * 100
    precisao_gatilho = (disseSerGatilho/totalGatilho) * 100

    return acuracia, backdoor_rate, outros_erros_rate, precisao_gatilho


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