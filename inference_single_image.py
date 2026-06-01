import os
import argparse
import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO

def count_screws(image_path, model_path="best.pt", conf_threshold=0.394):
    """
    Função de inferência projetada sob os padrões de ML Engineering:
    - Error handling robusto
    - Parametrização via CLI
    - Foco direto na métrica de negócio (Contagem)
    """
    
    # 1. Validação de Dados e Tratamento de Erros
    if not os.path.exists(image_path):
        print(f"❌ Erro: Imagem não encontrada no caminho especificado: '{image_path}'.")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ Erro: Modelo de rede neural não encontrado em: '{model_path}'.")
        print("💡 Dica: Certifique-se de que o treinamento foi concluído e o arquivo de pesos existe.")
        return

    print(f"🚀 Carregando modelo (Weights): {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Falha crítica ao carregar a arquitetura do modelo: {e}")
        return

    print(f"📸 Processando imagem: {image_path} (Confidence Threshold: {conf_threshold})")
    
    # 2. Execução da Inferência
    try:
        results = model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False,         # Vamos salvar manualmente para injetar o texto
            verbose=False       # Suprime logs extensos do YOLO
        )
    except Exception as e:
        print(f"❌ Ocorreu um erro durante o processamento da imagem: {e}")
        return

    # 3. Extração de Valor de Negócio e Edição Visual
    result = results[0]
    num_parafusos = len(result.boxes)
    
    # Extrair a matriz da imagem (numpy array) com as bounding boxes já desenhadas pelo YOLO
    plotted_img = result.plot()
    
    # Configuração do texto (Heads-Up Display) usando OpenCV
    texto = f"TOTAL: {num_parafusos} PARAFUSOS"
    fonte = cv2.FONT_HERSHEY_DUPLEX
    
    # Sombra preta para garantir legibilidade independentemente da cor do fundo da imagem
    cv2.putText(plotted_img, texto, (40, 80), fonte, 2.0, (0, 0, 0), 8, cv2.LINE_AA)
    # Texto principal em um tom de verde neon vibrante
    cv2.putText(plotted_img, texto, (40, 80), fonte, 2.0, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Salvar a imagem customizada
    save_dir = os.path.join("output", "inferencia_avulsa")
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, base_name)
    cv2.imwrite(save_path, plotted_img)

    # 4. Apresentação (Interface para o Usuário)
    print("\n" + "="*50)
    print("📊 RESULTADO OFICIAL DA CONTAGEM")
    print("="*50)
    print(f"🔩 Parafusos detectados: {num_parafusos}")
    print(f"📁 Imagem com HUD salva em: {os.path.abspath(save_dir)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Padrão de Engenharia de Software: Utilização de Argument Parser (CLI API)
    parser = argparse.ArgumentParser(description="Testador de Inferência (YOLO) para imagem avulsa.")
    
    parser.add_argument("-i", "--image", 
                        type=str, 
                        required=False, 
                        help="Caminho para a imagem de teste. Se omitido, abre janela de seleção.")
                        
    parser.add_argument("-m", "--model", 
                        type=str, 
                        default="best.pt", 
                        help="Caminho para os pesos do modelo YOLO (padrão: best.pt).")
                        
    parser.add_argument("-c", "--conf", 
                        type=float, 
                        default=0.25, 
                        help="Limiar de Confiança / Confidence Threshold (padrão: 0.25).")
    
    args = parser.parse_args()
    
    image_path = args.image
    
    # UX Enhancement: Abrir janela de seleção caso o caminho não seja passado via CLI
    if not image_path:
        print("🖥️ Abrindo janela para selecionar a imagem (Verifique sua barra de tarefas)...")
        root = tk.Tk()
        root.withdraw() # Esconde a janela principal vazia
        root.attributes('-topmost', True) # Traz a janela de arquivos para frente
        
        image_path = filedialog.askopenfilename(
            title="Selecione a foto dos parafusos",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()
        
        if not image_path:
            print("❌ Operação cancelada. Nenhuma imagem foi selecionada.")
            exit()
            
    count_screws(image_path, args.model, args.conf)
