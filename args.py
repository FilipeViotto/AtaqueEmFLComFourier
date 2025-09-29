class Args():
    def __init__(self):
        self.taxaDeAprendizadoDoServidor = 0.01
        self.num_atacante = 10
        self.num_cliente = 100
        self.selecionar = 10
        self.epocasLocais = 2
        self.batchsize = 64
        self.epoca = 300                # estabelecido pelo artigo
        self.lr = 0.1                   # estabelecido pelo artigo
        self.proporcaoAtaque = 0.15     # alterei, no artigo é 0.15
        self.gatilho = 2                # passaro
        self.alvo = 0                   # aviao
        self.beta = 0.2                 # estabelecido pelo artigo
        self.fatorLambida = 1
        self.reducao = 0.02
        self.segundaReducao = 0.004