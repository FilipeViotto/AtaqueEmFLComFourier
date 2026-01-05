import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

for comDefesa in [True, False]:
    for wm in [10,20,30,40,50]:
        nomeArquivo = os.path.join(f'relatorio/{'ATK'if comDefesa else 'NOATK'}')

        nomeRelatoriotxt = os.path.join(nomeArquivo, f'{wm}.txt')

        nomeRelatoriopng = os.path.join(nomeArquivo, f'{wm}.png')

        with open(nomeRelatoriotxt) as f:
            listaParaRelatorio = [float(linha.strip()) for linha in f if linha.strip()]
        
        plt.figure()
        plt.plot(listaParaRelatorio)
        plt.xlabel('epocas')
        plt.ylabel('menosAtacantes')
        plt.ylim(0,1.1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(nomeRelatoriopng)
        plt.close()