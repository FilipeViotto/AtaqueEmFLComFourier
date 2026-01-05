import torch
import copy
from args import Args


def agregaGrupos(grupos:list, modeloGlobal:torch.nn.Module):

    gruposAgregados = []
    for grupo in grupos:  # para cada grupo em grupos
        
        agregado = copy.deepcopy(modeloGlobal)  # separa um modelo para guardar a agregação desse grupo
        agregadoDict = agregado.state_dict()    # transforma o modelo em dicionario para inserir a soma dos pesos
        with torch.no_grad():
            for key in agregadoDict:    # para cada camada
                if agregadoDict[key].is_floating_point():
                    agregadoDict[key].zero_()   # zera os valores do gradiente para usar como somatório
        globalDict = modeloGlobal.state_dict()
        for modelo in grupo:    # para cada modelo do grupo
            dicionario = modelo.state_dict()    # transforma em dicionario
            
            with torch.no_grad():
                for key in globalDict.keys():   # percorre as camadas
                    if agregadoDict[key].is_floating_point():
                        agregadoDict[key] += torch.div(dicionario[key],len(grupo))    # soma no dicionario de agregação
                    else:
                        agregadoDict[key] = globalDict[key]
        agregado.load_state_dict(agregadoDict) # modelo agregado desse grupo
        gruposAgregados.append(agregado) # guarda o grupo
    return gruposAgregados

# envia os grupos para os clientes
# cada cliente testa e envia o F1-score

def selecionarNovoModeloGlobal(votos:dict, ListaModelos: list, args: Args):
    """recebe uma lista com os votos e a lista de modelos. retorna o modelo mais votado"""
    # pontuacoes = [voto['weighted'] for voto in votos]
    somaPontuacao = [0]*len(ListaModelos)
    votoModelo = [0]*len(ListaModelos)
    for cliente, voto in votos.items():
        votoModelo[voto.index(max(voto))]
        somaPontuacao = [somaPontuacao[i]+voto[i] for i in range(len(somaPontuacao))]
    
    somaPontuacao = [voto/args.num_cliente for voto in somaPontuacao]
  
    indiceModelo = votoModelo.index(max(votoModelo))
    return ListaModelos[indiceModelo].state_dict(), indiceModelo, somaPontuacao     # returna novo modelo global
