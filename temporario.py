import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
from sklearn.ensemble import IsolationForest

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = BasicBlock(64, 64, stride=1)
        self.block2 = BasicBlock(64, 128, stride=2)
        self.block3 = BasicBlock(128, 256, stride=2)
        self.block4 = BasicBlock(256, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Função de Defesa ANODYNE Completa ---

def defesa(modelos: list, d_sub_vector: int = 1000, historico_updates: dict = None):
    """
    Implementa as etapas de subdivisão, geração de métricas e detecção da defesa ANODYNE.
    
    Args:
        modelos (list): Uma lista de modelos PyTorch (um de cada cliente).
        d_sub_vector (int): O tamanho padrão de cada sub-vetor.
        historico_updates (dict): Dicionário contendo os updates da rodada anterior.

    Returns:
        list: Uma lista com os índices dos clientes classificados como benignos.
    """
    if not modelos:
        print("A lista de modelos está vazia.")
        return []

    num_clientes = len(modelos)

    # --- ETAPA 1: Coleta e Organização dos Parâmetros ---
    updates = {}
    for i, modelo in enumerate(modelos):
        for nomeCamada, camada in modelo.named_modules():
            if isinstance(camada, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                for nomeParametro, parametro in camada.named_parameters():
                    full_param_name = f"{nomeCamada}.{nomeParametro}"
                    if full_param_name not in updates:
                        updates[full_param_name] = [None] * num_clientes
                    updates[full_param_name][i] = parametro.data.flatten()
    
    print(f"--- Coleta finalizada. {len(updates)} grupos de parâmetros a serem analisados. ---\n")

    # --- ETAPA 2: Subdivisão e Cálculo das Métricas ---
    # Estrutura: all_sub_vector_features[id_sub_vetor_global] = [vetor_features_cliente1, ...]
    all_sub_vector_features = []
    
    # Mapeamento para saber a qual cliente cada vetor de feature pertence
    sub_vector_client_map = []
    
    for param_name, param_vectors_list in updates.items():
        if not param_vectors_list or param_vectors_list[0] is None:
            continue
            
        total_len = param_vectors_list[0].numel()
        num_sub_vectors = (total_len + d_sub_vector - 1) // d_sub_vector

        for p in range(num_sub_vectors):
            start_idx = p * d_sub_vector
            end_idx = min((p + 1) * d_sub_vector, total_len)
            
            sub_vectors_clients = [vec[start_idx:end_idx] for vec in param_vectors_list]
            sub_vectors_tensor = torch.stack(sub_vectors_clients)
            
            # --- Cálculo das Métricas para cada cliente ---
            features_for_this_subvector = []
            for i in range(num_clientes):
                # Métricas Espaciais: distância média para os outros clientes
                dist_euclidiana = torch.mean(torch.norm(sub_vectors_tensor[i] - torch.cat((sub_vectors_tensor[:i], sub_vectors_tensor[i+1:])), p=2, dim=1))
                dist_manhattan = torch.mean(torch.norm(sub_vectors_tensor[i] - torch.cat((sub_vectors_tensor[:i], sub_vectors_tensor[i+1:])), p=1, dim=1))
                cosine_sim = torch.mean(F.cosine_similarity(sub_vectors_tensor[i].unsqueeze(0), torch.cat((sub_vectors_tensor[:i], sub_vectors_tensor[i+1:]))))

                # Métricas Temporais (se houver histórico)
                dist_temporal = 0.0
                if historico_updates and param_name in historico_updates:
                    sub_vetor_anterior = historico_updates[param_name][i][start_idx:end_idx]
                    dist_temporal = torch.norm(sub_vectors_clients[i] - sub_vetor_anterior, p=2)



                # Criar o vetor de características ("dossiê") para este sub-vetor deste cliente
                feature_vector = [
                    dist_euclidiana.item(), 
                    dist_manhattan.item(),
                    cosine_sim.item(),
                    dist_temporal.item() if isinstance(dist_temporal, torch.Tensor) else dist_temporal
                ]
                features_for_this_subvector.append(feature_vector)
                sub_vector_client_map.append(i) # Guarda o ID do cliente

            all_sub_vector_features.extend(features_for_this_subvector)

    print(f"--- Geração de Métricas finalizada. {len(all_sub_vector_features)} vetores de características criados. ---\n")

    # --- ETAPA 3: Detecção de Gradientes Maliciosos ---
    
    # 1. Converter para numpy e alimentar o Isolation Forest
    X = np.array(all_sub_vector_features)
    if not np.any(np.isfinite(X)):
        print("Erro: Vetor de características contém valores não finitos (NaN ou Inf).")
        # Retorna todos os clientes como benignos em caso de erro
        return list(range(num_clientes))

    clf = IsolationForest(contamination='auto', random_state=42)
    clf.fit(X)
    
    # Scores de anomalia (quanto menor, mais anômalo)
    anomaly_scores = clf.decision_function(X)
    
    # 2. Votação Suave (Soft Voting)
    client_scores = np.zeros(num_clientes)
    for i, score in enumerate(anomaly_scores):
        client_id = sub_vector_client_map[i]
        client_scores[client_id] += score
        
    print("--- Scores de Anomalia por Cliente (quanto menor, mais suspeito) ---")
    for i, score in enumerate(client_scores):
        print(f"Cliente {i}: Score Agregado = {score:.2f}")

    # 3. Decisão: Consideramos maliciosos os clientes com score abaixo da mediana, por exemplo
    # Uma estratégia simples é remover os 'n' piores.
    # Outra é usar um desvio padrão da média, ou um threshold fixo.
    # Aqui, vamos usar a mediana como um threshold dinâmico.
    median_score = np.median(client_scores)
    benign_clients_indices = [i for i, score in enumerate(client_scores) if score >= median_score]
    
    print(f"\nMediana dos scores: {median_score:.2f}")
    print(f"Clientes benignos identificados (índices): {benign_clients_indices}")

    return benign_clients_indices, updates # Retorna também os updates para a próxima rodada

if __name__ == '__main__':
    num_clientes = 10
    print(f"Iniciando simulação com {num_clientes} clientes.")
    
    # 1. Simular modelos da rodada anterior (para histórico)
    modelos_anteriores = [ResNet18(num_classes=10) for _ in range(num_clientes)]
    
    # 2. Criar o dicionário de histórico a partir desses modelos
    historico = {}
    for i, modelo in enumerate(modelos_anteriores):
        for nomeCamada, camada in modelo.named_modules():
            if isinstance(camada, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                for nomeParametro, parametro in camada.named_parameters():
                    full_param_name = f"{nomeCamada}.{nomeParametro}"
                    if full_param_name not in historico:
                        historico[full_param_name] = [None] * num_clientes
                    historico[full_param_name][i] = parametro.data.flatten()
    
    # 3. Simular modelos da rodada ATUAL
    modelos_atuais = [ResNet18(num_classes=10) for _ in range(num_clientes)]
    for modelo in modelos_atuais:
        for p in modelo.parameters():
            p.data += torch.randn_like(p.data) * 0.01 # Treinamento benigno normal

    # 4. SIMULAR UM ATAQUE: Fazer um cliente (ex: cliente 3) ter um update muito diferente
    cliente_malicioso_idx = 3
    print(f"\n!!! Simulando um ataque do cliente {cliente_malicioso_idx} !!!\n")
    for p in modelos_atuais[cliente_malicioso_idx].parameters():
        p.data *= 5 # Amplifica os pesos drasticamente
        p.data += torch.randn_like(p.data) * 2 # Adiciona ruído de alta magnitude

    # 5. Executar a função de defesa completa
    clientes_benignos, _ = defesa(modelos_atuais, d_sub_vector=5000, historico_updates=historico)
