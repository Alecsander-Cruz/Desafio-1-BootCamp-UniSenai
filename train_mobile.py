import os
from ultralytics import YOLO

def main():
    print("Carregando o modelo YOLO26 Nano (O mais novo modelo NMS-Free focado em Mobile/Edge)...")
    # Carregando modelo base nano para detecção de objetos (Bounding Boxes)
    model = YOLO("yolo26n.pt")

    try:
        print("Iniciando o treinamento para contagem de peças...")
        results = model.train(
            data="data.yaml",     
            cfg="aug.yaml",
            epochs=150,
            patience=50,          
            imgsz=1024,           # Resolução alta para não perder os parafusos menores
            batch=4,              # O lote ideal para não estourar a memória na resolução 1024
            project="output",    
            name="yolo26n_parafusos_mobile",
            plots=True,           
            workers=8,            
            cache=False,          
            device=0              
        )
        print("Treinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"\nOcorreu um erro durante o treino: {e}")
        print("Salvando backup parcial...")
        
    finally:
        print("\nZipando os resultados e enviando para o Google Drive...")
        import shutil
        import glob
        
        try:
            # Buscar pastas criadas na raiz ou dentro de runs/detect/
            pastas = glob.glob("output/*") + glob.glob("runs/detect/output/*") + glob.glob("runs/detect/*")
            if pastas:
                pasta_recente = max(pastas, key=os.path.getmtime)
                print(f"Compactando a pasta: {pasta_recente}")
                
                # Nome do arquivo zipado
                shutil.make_archive("resultados_mobile_parafusos", 'zip', pasta_recente)
                
                # Cópia para o Drive montado no Colab
                destino = "/content/drive/MyDrive/resultados_mobile_parafusos.zip"
                shutil.copy("resultados_mobile_parafusos.zip", destino)
                print(f"Backup copiado com sucesso para: {destino}")
            else:
                print("Nenhuma pasta de output foi encontrada para backup.")
                
        except Exception as bkp_err:
            print(f"Erro ao copiar para o Drive: {bkp_err}")
            print("Verifique se o Google Drive foi montado corretamente no Colab.")
            print("Abortando o desligamento da máquina para proteger os arquivos...")
            return 
        
        print("\nDesligando a instância para economizar créditos do Colab...")
        try:
            from google.colab import runtime
            runtime.unassign()
        except ImportError:
            print("(Execução fora do Colab: Desligamento ignorado)")

if __name__ == "__main__":
    main()
