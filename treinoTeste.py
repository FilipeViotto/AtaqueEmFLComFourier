import torch.nn as nn
import torch
from args import Args
args = Args()
from modelos import ModeloMnist
device = torch.device('cpu')



def treinar(modelos, train_list, listaOptim:list):
    criterion = nn.CrossEntropyLoss()
    for modelo, conjuntoDeTreino, optim in zip(modelos, train_list, listaOptim):
        modelo.train()
        for imagem, rotulo in conjuntoDeTreino:
            optim.zero_grad()
            imagem, rotulo = imagem.to(device), rotulo.to(device)
            out = modelo(imagem)
            loss = criterion(out, rotulo)
            loss.backward()
            optim.step()


def testar(modelo: ModeloMnist, teste):
    modelo.eval()
    totalUm = 0
    totalSete = 0
    erroPorBack = 0
    outrosErros = 0
    acertos_totais = 0
    total_amostras = 0
    disseSerSete = 0

    with torch.no_grad():
        # O loop externo pega um BATCH de imagens e rótulos
        for x, y in teste:
            x = x.to(device)
            y = y.to(device)

            out = modelo(x)
            _, predictions = out.max(1)

            # --- CORREÇÃO PRINCIPAL: Loop Interno ---
            # Itera sobre cada item DENTRO do batch
            for i in range(len(y)):
                true_label = y[i]
                predLabel = predictions[i]

                # Agora as comparações são feitas com valores únicos
                if true_label == predLabel:
                    acertos_totais += 1

                if true_label == 1:
                    totalUm += 1
                    if predLabel == 7:
                        erroPorBack += 1
                    elif predLabel != 1:
                        outrosErros += 1
                
                elif true_label == 7:
                    totalSete += 1
                
                if predLabel == 7:
                    disseSerSete += 1
            
            total_amostras += y.size(0)

    # Evitar divisão por zero se não houver amostras
    if totalUm == 0: totalUm = 1
    if disseSerSete == 0: disseSerSete = 1
    if total_amostras == 0: total_amostras = 1

    # Cálculos de acurácia corrigidos
    acuracia = (acertos_totais / total_amostras) * 100
    backdoor_rate = (erroPorBack / totalUm) * 100
    outros_erros_rate = (outrosErros / totalUm) * 100
    # Precisão para a classe 7: dos que o modelo disse ser 7, quantos realmente eram 7
    precisao_sete = (y[predictions == 7] == 7).sum().item() / disseSerSete * 100

    return acuracia, backdoor_rate, outros_erros_rate, precisao_sete