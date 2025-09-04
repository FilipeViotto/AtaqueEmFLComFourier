import torch
from args import Args
args = Args()





def avg(models):

    agregado = models[0].state_dict()
    for modelo in models[1:]:
        dicionario = modelo.state_dict()
        for key in dicionario.keys():
            agregado[key] += dicionario[key]
    for key in agregado.keys():
        agregado[key] = torch.div(agregado[key], args.num_cliente)
    
    for i, modelo in enumerate(models):
        models[i].load_state_dict(agregado)