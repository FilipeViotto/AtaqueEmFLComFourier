import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os

seu_email = "soparasimular@gmail.com"
senha_app = "vkzz flgs lgei pimm"
destinatario = "filipe.filipe2009@gmail.com"

servidor_smtp = "smtp.gmail.com"
porta = 465

# A função agora aceita o caminho da imagem como um argumento
def enviarEmail(texto, caminho_imagem):
    assunto = "Relatorio de treino com modelo simplificado"
    corpo_email = f"{texto}"
    
    msg = MIMEMultipart()
    msg['From'] = seu_email
    msg['To'] = destinatario
    msg['Subject'] = assunto
    
    # Anexa o corpo do email (o texto)
    msg.attach(MIMEText(corpo_email, 'plain'))
    
    # --- Seção para adicionar a imagem ---
    try:
        # 2. Abrir a imagem em modo de leitura binária ('rb')
        with open(caminho_imagem, 'rb') as f:
            # Ler os dados da imagem
            img_data = f.read()
            # 3. Criar o anexo MIMEImage
            # Usamos os.path.basename para extrair apenas o nome do arquivo do caminho
            imagem = MIMEImage(img_data, name=os.path.basename(caminho_imagem))
            # 4. Anexar a imagem à mensagem principal
            msg.attach(imagem)
            
    except FileNotFoundError:
        print(f"Erro: O arquivo de imagem não foi encontrado em '{caminho_imagem}'")
        return # Para a execução se o arquivo não existir

    # --- Fim da seção da imagem ---

    print("Enviando email...")
    try:
        with smtplib.SMTP_SSL(servidor_smtp, porta) as server:
            server.login(seu_email, senha_app)
            server.sendmail(seu_email, destinatario, msg.as_string())
        print("Email enviado com sucesso!")
    except Exception as e:
        print(f"Falha ao enviar o email: {e}")