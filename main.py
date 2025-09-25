import matplotlib.pyplot as plt
import numpy as np
import torch
from args import Args
from torch.utils.data import DataLoader, random_split
import down_datas
args = Args()
from modelos import Modelo, ModeloCifar10, ModeloCifar10_Revisado
import torch.optim as optim
device = torch.device('cpu')
from agregacoes import avg
from treinoTeste import testar, treinar
from ataque import ataqueCifar, ataqueMnist

def pegar_dados_iid():
    # Baixa os dados de treino
    train_data, test_data, classes = down_datas.down_cifar()

    num_exemplos = len(train_data)
    exemplo_por_cliente = num_exemplos//args.num_cliente
    tamanho_dataset = [exemplo_por_cliente] * args.num_cliente
    
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    client_datasets = random_split(train_data, tamanho_dataset)
    
    generator = torch.Generator().manual_seed(SEED)
    lista_dataloaders = []
    for dataset in client_datasets:
        loader = DataLoader(dataset,args.batchsize,True,generator=generator)
        lista_dataloaders.append(loader)

    torch.manual_seed(SEED)
    teste = random_split(test_data, [10000])
    loaderTest = DataLoader(teste[0], args.batchsize, False)
    
    return loaderTest, lista_dataloaders, classes


testSet, trainSetList, classes = pegar_dados_iid()

camada = None
for dado in trainSetList:
    for imagem, _ in dado:
        camada = imagem.shape[1]
        break
        


listaDeModelos = [ModeloCifar10().to(device) for i in range(args.num_cliente)]
listaOptim = [optim.Adam(modelo.parameters(), lr=args.lr,) for modelo in listaDeModelos]
listaAcuracia = []
errosCertos = []
errosErrados = []
precisaoGatilho = []

ataqueCifar(trainSetList)

try:
    for epoca in range(args.epoca):
        print(f'\nEp {epoca}')
        
        treinar(listaDeModelos, trainSetList, listaOptim)
        avg(listaDeModelos)
        acc, backdoor, naoBackdoor, GatilhoEGatilho = testar(listaDeModelos[0], testSet, device, classes)
        print(f'acc:{acc}, backdoor:{backdoor}, GatilhoCerto:{GatilhoEGatilho}, nãoBackdoor:{naoBackdoor}')

        listaAcuracia.append(acc)
        errosCertos.append(backdoor)
        errosErrados.append(naoBackdoor)
        precisaoGatilho.append(GatilhoEGatilho)

finally:
    plt.plot(range(len(listaAcuracia)), listaAcuracia)
    plt.title('variacao da acuracia')
    plt.xlabel('epocas')
    plt.ylabel('acuracia')
    plt.savefig('simulacao/variacao_acuracia.png')
    plt.close()

    plt.plot(range(len(errosCertos)), errosCertos)
    plt.title('backdoor atingiu o alvo')
    plt.xlabel('epocas')
    plt.ylabel('erros acertados')
    plt.savefig('simulacao/backdoor_sucesso.png')
    plt.close()

    plt.plot(range(len(errosErrados)), errosErrados)
    plt.title('erros nao atingiu o alvo')
    plt.xlabel('epocas')
    plt.ylabel('erros errados')
    plt.savefig('simulacao/backdoor_fracasso.png')
    plt.close()

    plt.plot(range(len(precisaoGatilho)), precisaoGatilho)
    plt.title('Precisão sobre o gatilho')
    plt.xlabel('Epocas')
    plt.ylabel('Precisão sobre o gatilho')
    plt.savefig('simulacao/precisao_gatilho.png')
    plt.close()



# print("Dados de treino baixados:", train_data)
# print("Dados de teste baixados:", test_data)


# imagem, rotulo = train_data[0]
# imagem_sq = imagem.squeeze()
# img = plt.imshow(imagem_sq, cmap='gray')
# plt.savefig('imagem_retirada.png')

# amp = amplitude(imagem)
