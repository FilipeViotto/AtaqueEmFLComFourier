
import down_datas
from ataque import get_trigger_amplitudes, PoisonedDataset
from args import Args
args = Args()
import torch 
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import DataLoader, random_split, Subset

def pegar_dados_iid():
    train_data, test_data, classes = down_datas.down_cifar()
    trigger_amplitudes = get_trigger_amplitudes(train_data, args)

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
    dataset_teste_completo = DataLoader(test_data, args.batchsize, shuffle=False)
    
    num_exemplos_teste = len(test_data)
    exemplo_por_cliente_teste = num_exemplos_teste // args.num_cliente
    tamanho_dataset_teste = [exemplo_por_cliente_teste] * (args.num_cliente - 1)
    tamanho_dataset_teste.append(num_exemplos_teste - sum(tamanho_dataset_teste))
    
    dataset_splits_teste = random_split(test_data, tamanho_dataset_teste, generator=generator)
    lista_dataloaders_teste = [DataLoader(ds, args.batchsize, shuffle=False) for ds in dataset_splits_teste]

    return dataset_teste_completo, lista_dataloaders_teste, lista_dataloaders_treino, classes