import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import IsolationForest

def defesa(modelos: list, historico=None):
    numeroClientes = len(modelos)
    divisao = 1000  # Tamanho padrão do sub-vetor
    atualizacoes = {} # armazena os subvetores da epoca atual
    
    for i, modelo in enumerate(modelos):    # para cada modelo
        for nomeCamada, camada in modelo.named_modules():   # para cada camada do modelo
            if isinstance(camada, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                for nomeParametro, parametro in camada.named_parameters():  # percorre a camada 
                    flattendParametros = parametro.data.flatten()   # achata os parametros
                    nomeCompleto = f"{nomeCamada}.{nomeParametro}"  # identifica a camada
                    
                    if nomeCompleto not in atualizacoes:
                        atualizacoes[nomeCompleto] = [None] * numeroClientes    
                    atualizacoes[nomeCompleto][i] = flattendParametros      # insere atualização atual
    
    all_sub_vector_features = []
    sub_vector_client_map = []

    for nomeParametro, listaVetorParametros in atualizacoes.items():        # para cada parametro 
        if not listaVetorParametros or listaVetorParametros[0] is None:
            continue
        
        tamanho = listaVetorParametros[0].numel()
        numeroSubvetores = (tamanho + divisao - 1) // divisao
        
        for parte in range(numeroSubvetores):
            indiceInicial = parte * divisao     # inicio do subetor
            indiceFinal = min((parte + 1) * divisao, tamanho)   # fim do subvetor
            subVetores = [vet[indiceInicial:indiceFinal] for vet in listaVetorParametros]   # percorre os parametos de cada cliente e seleciona um determinado trecho que será um subvetor 

            if any(s.numel() == 0 for s in subVetores):
                continue
            
            subVetoresTensor = torch.stack(subVetores)      # empilha esses subvetores em um tensor
            
            for i in range(numeroClientes): # para cada cliente
                outros_vetores = torch.cat((subVetoresTensor[:i], subVetoresTensor[i+1:]))  # percorre todos os outros subvetores, um de cada vez
                # compara o subvetor atual com os outros
                distEuclidiana = torch.mean(torch.norm(subVetoresTensor[i] - outros_vetores, p=2, dim=1))
                distManhattan = torch.mean(torch.norm(subVetoresTensor[i] - outros_vetores, p=1, dim=1))
                cosseno = torch.mean(F.cosine_similarity(subVetoresTensor[i].unsqueeze(0), outros_vetores))

                # Métricas Temporais (distância do cliente 'i' para seu histórico)
                euclidianaTemporal = 0.0
                mahattanTemporal = 0.0
                cossenoTemporal = 0.0
                if historico and nomeParametro in historico and historico[nomeParametro][i] is not None:
                    if len(historico[nomeParametro][i]) > indiceInicial:
                        subVetorAnterior = historico[nomeParametro][i][indiceInicial:indiceFinal]
                        if subVetores[i].shape == subVetorAnterior.shape:
                            euclidianaTemporal = torch.norm(subVetores[i] - subVetorAnterior, p=2)
                            mahattanTemporal = torch.norm(subVetores[i] - subVetorAnterior, p=1)
                            cossenoTemporal = F.cosine_similarity(subVetores[i], subVetorAnterior, dim=0)

                vetor_features = [      # lista com todas as comparações
                    distEuclidiana.item(),
                    distManhattan.item(),
                    cosseno.item(),
                    euclidianaTemporal.item() if isinstance(euclidianaTemporal, torch.Tensor) else euclidianaTemporal,
                    mahattanTemporal.item() if isinstance(mahattanTemporal, torch.Tensor) else mahattanTemporal,
                    cossenoTemporal.item() if isinstance(cossenoTemporal, torch.Tensor) else cossenoTemporal
                ]
                all_sub_vector_features.append(vetor_features) # vai juntando as comparações
                sub_vector_client_map.append(i) # Mapeia cada feature de volta a um cliente

    
    if not all_sub_vector_features:
        return atualizacoes, list(range(numeroClientes))

    X = np.array(all_sub_vector_features)
    
    clf = IsolationForest(contamination='auto', random_state=42)
    clf.fit(X)  
    anomaly_scores = clf.decision_function(X) # Scores mais baixos são mais anômalos


    pontuacao_total_cliente = np.zeros(numeroClientes)
    for idx, score in enumerate(anomaly_scores):
        clienteId = sub_vector_client_map[idx]
        pontuacao_total_cliente[clienteId] += score

    limiar = np.median(pontuacao_total_cliente) # foi usado a mediana
    
    # scores mais altos são mais "normais" (benignos)
    benignos = [i for i, score in enumerate(pontuacao_total_cliente) if score >= limiar]

    print(f'Clientes benignos identificados: {benignos}')

    return atualizacoes, benignos
