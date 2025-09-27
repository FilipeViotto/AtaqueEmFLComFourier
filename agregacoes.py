import copy
import torch
from args import Args

def avg(models, modelo_gobal_atual, selecionados, args: Args):

    agregado = copy.deepcopy(modelo_gobal_atual)

    agregado_dic = agregado.state_dict()
    with torch.no_grad():
        for key in agregado_dic:
            if agregado_dic[key].is_floating_point():
                agregado_dic[key].zero_()
    
    dicionario_global = modelo_gobal_atual.state_dict()
    for i in selecionados:
        modelo = models[i]
        dicionario = modelo.state_dict()
        for key in dicionario_global.keys():
            if agregado_dic[key].is_floating_point():
                    agregado_dic[key] += (dicionario[key] - dicionario_global[key])*args.fatorLambida if i < args.num_atacante else (dicionario[key] - dicionario_global[key])
                    # agregado_dic[key] += dicionario[key]
    
    for key in agregado_dic.keys():
        if agregado_dic[key].is_floating_point():

            # agregado_dic[key] = torch.div(agregado_dic[key], args.selecionar)
            # agregado_dic[key] = dicionario_global[key] + media_deltas
            media_deltas = torch.div(agregado_dic[key], len(selecionados))
            agregado_dic[key] = dicionario_global[key] + media_deltas * args.taxaDeAprendizadoDoServidor
        else:
            agregado_dic[key] = dicionario_global[key]
        
    modelo_gobal_atual.load_state_dict(agregado_dic)
    
    for i, modelo in enumerate(models):
        models[i].load_state_dict(agregado_dic)


def avg_padrao(models: list, modelo_global_atual, selecionados):
    """
    Implementa o algoritmo de agregação FedAvg padrão.
    Calcula a média ponderada dos pesos dos modelos dos clientes selecionados
    e atualiza o modelo global com essa média.
    """
    
    # Pega o state_dict do modelo global para usar como estrutura
    global_dict = modelo_global_atual.state_dict()
    
    # Zera o dicionário global para usá-lo como um acumulador
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key])

    # ### MUDANÇA CRÍTICA ###
    # Soma os state_dicts de todos os modelos de clientes selecionados
    for i in selecionados:
        client_dict = models[i].state_dict()
        for key in global_dict.keys():
            global_dict[key] += client_dict[key]
            
    # Calcula a média dividindo pela quantidade de clientes selecionados
    for key in global_dict.keys():
        global_dict[key] = torch.div(global_dict[key], len(selecionados))

    # Carrega a nova média de pesos no modelo global
    modelo_global_atual.load_state_dict(global_dict)
    
    # Distribui o novo modelo global atualizado para todos os clientes
    # (No FL real, isso aconteceria na próxima rodada de treino)
    for modelo in models:
        modelo.load_state_dict(global_dict)