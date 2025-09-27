import copy
import torch
from args import Args
args = Args()

def avg(models, modelo_gobal_atual, selecionados):

    agregado = copy.deepcopy(modelo_gobal_atual)

    agregado_dic = agregado.state_dict()
    with torch.no_grad():
        for key in agregado_dic:
            if agregado_dic[key].is_floating_point():
                agregado_dic[key].zero_()
    
    dicionario_global = modelo_gobal_atual.state_dict()
    for modelo in [models[i] for i in selecionados]:
        dicionario = modelo.state_dict()
        for key in dicionario_global.keys():
            if agregado_dic[key].is_floating_point():
                    agregado_dic[key] += (dicionario[key] - dicionario_global[key])
    
    for key in agregado_dic.keys():
        if agregado_dic[key].is_floating_point():
            media_deltas = torch.div(agregado_dic[key], args.selecionar)
            agregado_dic[key] = dicionario_global[key] + media_deltas * args.taxaDeAprendizadoDoServidor
        else:
            agregado_dic[key] = dicionario_global[key]
        
    modelo_gobal_atual.load_state_dict(agregado_dic)
    
    for i, modelo in enumerate(models):
        models[i].load_state_dict(agregado_dic)