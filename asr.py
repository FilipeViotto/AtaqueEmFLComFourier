import torch

def calcular_asr(modelo, poisoned_test_set, target_label, device = torch.device('cpu')):
    modelo.eval()
    sucessos_ataque = 0
    total_amostras = 0

    with torch.no_grad():
        for x, y in poisoned_test_set:
            x = x.to(device)
            # y (rótulo verdadeiro) não é usado para o acerto, mas para o total
            
            out = modelo(x)
            _, predictions = out.max(1)

            # Conta quantas predições foram iguais ao rótulo alvo do ataque
            sucessos_ataque += (predictions == target_label).sum().item()
            total_amostras += x.size(0)

    # Evitar divisão por zero
    if total_amostras == 0:
        return 0.0

    asr = (sucessos_ataque / total_amostras) * 100
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    return asr