import matplotlib.pyplot as plt
import numpy as np
import torch
from args import Args
from torch.utils.data import DataLoader, random_split
import down_datas
args = Args()
from modelos import ModeloMnist
import torch.optim as optim
device = torch.device('cpu')
from agregacoes import avg
from treinoTeste import testar, treinar
from ataque import preparaMinstAtaque

def pegar_dados_iid():
    # Baixa os dados de treino
    train_data, test_data, classes = down_datas.down_mnist()

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

    # Baixa os dados de teste
    SEED = 42
    torch.manual_seed(SEED)
    teste = random_split(test_data, [10000])
    loaderTest = DataLoader(teste[0], args.batchsize, False)
    
    return loaderTest, lista_dataloaders


testSet, trainSetList = pegar_dados_iid()


listaDeModelos = [ModeloMnist().to(device) for i in range(args.num_cliente)]
listaOptim = [optim.Adam(modelo.parameters(), lr=args.lr,) for modelo in listaDeModelos]
listaAcuracia = []
errosCertos = []
errosErrados = []
setes = []

preparaMinstAtaque(trainSetList)
for epoca in range(args.epoca):
    print(f'epoca {epoca}')
    
    treinar(listaDeModelos, trainSetList, listaOptim)
    avg(listaDeModelos)
    acc, backdoor, naoBackdoor, seteEraSete = testar(listaDeModelos[0], testSet)
    print(f'acc {acc}')

    listaAcuracia.append(acc)
    errosCertos.append(backdoor)
    errosErrados.append(naoBackdoor)
    setes.append(seteEraSete)


plt.plot(range(args.epoca), listaAcuracia)
plt.title('variacao da acuracia')
plt.xlabel('epocas')
plt.ylabel('acuracia')
plt.savefig('variacao_acuracia.png')
plt.close()

plt.plot(range(args.epoca), errosCertos)
plt.title('backdoor atingiu o alvo')
plt.xlabel('epocas')
plt.ylabel('erros acertados')
plt.savefig('backdoor_sucesso.png')
plt.close()

plt.plot(range(args.epoca), errosErrados)
plt.title('erros nao atingiu o alvo')
plt.xlabel('epocas')
plt.ylabel('erros errados')
plt.savefig('backdoor_fracasso.png')
plt.close()

plt.plot(range(args.epoca), setes)
plt.title('sete era sete')
plt.xlabel('epocas')
plt.ylabel('era sete')
plt.savefig('sete_era_sete.png')
plt.close()





 



# print("Dados de treino baixados:", train_data)
# print("Dados de teste baixados:", test_data)


# imagem, rotulo = train_data[0]
# imagem_sq = imagem.squeeze()
# img = plt.imshow(imagem_sq, cmap='gray')
# plt.savefig('imagem_retirada.png')

# amp = amplitude(imagem)
