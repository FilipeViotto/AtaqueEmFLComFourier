import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
from sklearn.ensemble import IsolationForest

def defesa(modelos: list, historico = None):
    
    numeroClientes = len(modelos)
    divisao = 1000
    atualizacoes = {}
    for i, modelo in enumerate(modelos):  # pega um modelo
        for nomeCamada, camada in modelo.named_modules():       # percorre as camadas dos modelos
            if isinstance(camada, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                for nomeParametro, parametro in camada.named_parameters():          # percorre todos parametros do modelo
                    flattendParametros = parametro.data.flatten()           # transforma em lista
                    nomeCompleto = f"{nomeCamada}.{nomeParametro}"          # identificador
                    if nomeCompleto not in atualizacoes:                    # se identificador não tiver registrado, cria uma lista nova
                        atualizacoes[nomeCompleto] = [None] * len(modelos) # substituir len() pela quantidade de modelos selecionados
                    atualizacoes[nomeCompleto][i] = flattendParametros # atualizações é um dicionario de listas de parametros
    
    all_sub_vector_features = []
    sub_vector_client_map = []

    for nomeParametro, listaVetorParametros in atualizacoes.items():        # pega uma lista de parametros
        if not listaVetorParametros or listaVetorParametros[0] is None:
            continue
        
        tamanho = listaVetorParametros[0].numel()  # obtem o tamanho da lista
        numeroSubvetores = (tamanho + divisao - 1) // divisao               # numero de subvetores que devem ser feitos
        for parte in range(numeroSubvetores): # para cada parte do subvetor
            indiceInicial = parte * divisao         # calcula posição inicial da parte
            indiceFinal = min((parte + 1) * divisao, tamanho)
            subVetores = [vet[indiceInicial:indiceFinal] for vet in listaVetorParametros] # recorta a lista de vetor
            subVetoresTensor = torch.stack(subVetores) # empilha o subvetor


            featuresParaSubvetor = []
            for i in range(numeroClientes):
            #Distância Euclidiana
                distEuclidiana = torch.mean(torch.norm(subVetoresTensor[i] - torch.cat((subVetoresTensor[:i],subVetoresTensor[i+1:])), p=2, dim=1))
            # Manhattan
                distManhattan = torch.mean(torch.norm(subVetoresTensor[i] - torch.cat((subVetoresTensor[:i], subVetoresTensor[i+1:])),p=1, dim=1))
            # Cosseno
                cosseno = torch.mean(F.cosine_similarity(subVetoresTensor[i].unsqueeze(0),torch.cat((subVetoresTensor[i+1:]))))

                distanciaTemporal = 0
                if historico and nomeParametro in historico: # verifico se existe histórico, na primeira epoca não tem, mas na segunda já tem.
                    subVetorAnterior = historico[nomeParametro][i][indiceInicial:indiceFinal]
                    distanciaTemporal = torch.norm(subVetores[i]-subVetorAnterior, p=2)

                vetor = [
                    distEuclidiana.item(),
                    distManhattan.item(),
                    cosseno.item(),
                    distanciaTemporal.item() if isinstance(distanciaTemporal, torch.Tensor) else distanciaTemporal
                ]
                featuresParaSubvetor.append(vetor)
                sub_vector_client_map.append(i)
            
            all_sub_vector_features.extend(featuresParaSubvetor)

            X = np.array(all_sub_vector_features)
            if not np.any(np.isfinite(X)):
                return list(range(numeroClientes))
            
            clf = IsolationForest(contamination='auto', random_state=42)
            clf.fit(X)
            anomaly_scores = clf.decision_function(X)

            pontuacao = np.zeros(numeroClientes)
            for i, score in enumerate(anomaly_scores):
                clienteId = sub_vector_client_map[i]
                pontuacao[clienteId] += score
            
            pontuacaoMedia = np.median(pontuacao)
            benignos = [i for i, score in enumerate (pontuacao) if score >= pontuacaoMedia]

            return [modelos[a] for a in benignos], atualizacoes