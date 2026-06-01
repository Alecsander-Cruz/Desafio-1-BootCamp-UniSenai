import os
from ultralytics import YOLO

def main():
    # Caminho onde o YOLO salva o melhor modelo treinado (conforme o train_mobile.py)
    caminho_modelo = "best.pt"
    
    if not os.path.exists(caminho_modelo):
        print(f"Erro: O modelo treinado não foi encontrado em '{caminho_modelo}'.")
        print("Dica: Certifique-se de que o treinamento no Colab gerou esta pasta ou ajuste o caminho.")
        return
        
    print(f"Carregando o modelo treinado: {caminho_modelo}")
    model = YOLO(caminho_modelo)
    
    # ---------------------------------------------------------
    # 1. AVALIAÇÃO TÉCNICA QUANTITATIVA (Métricas do ML Engineer)
    # ---------------------------------------------------------
    print("\n[1/2] Rodando Validação Formal (Métricas Técnicas)...")
    print("Isso vai avaliar o mAP, Precision e Recall no conjunto de teste.")
    try:
        # split="test" busca a pasta 'test' configurada no seu data.yaml
        metrics = model.val(
            data="data.yaml", 
            split="test", 
            project="output", 
            name="validacao_tecnica_parafusos"
        )
        print("\nResultados Técnicos:")
        print(f"  - mAP50-95 (Qualidade Geral): {metrics.box.map:.4f}")
        print(f"  - mAP50 (Precisão padrão):    {metrics.box.map50:.4f}")
        print(f"  - Precision (Precisão):       {metrics.box.p.mean():.4f}")
        print(f"  - Recall (Revocação):         {metrics.box.r.mean():.4f}")
    except Exception as e:
        print(f"Não foi possível rodar a validação formal. Verifique o data.yaml: {e}")

    # ---------------------------------------------------------
    # 2. AVALIAÇÃO DE NEGÓCIO E QUALITATIVA (Contagem Real)
    # ---------------------------------------------------------
    print("\n[2/2] Rodando Inferência Visual (Testando em imagens reais)...")
    pasta_teste = "dataset_split/test/images" # Atualizado para o novo split do dataset
    
    if os.path.exists(pasta_teste):
        # Pegar apenas algumas imagens para testar rapidamente (ex: as 5 primeiras)
        imagens = [os.path.join(pasta_teste, img) for img in os.listdir(pasta_teste) if img.endswith(('.jpg', '.png', '.jpeg'))][:]
        
        if imagens:
            # Roda a predição. save=True desenha as caixas e salva as imagens.
            # conf=0.25 é o limite de confiança. Ignora predições com menos de 25% de certeza.
            resultados = model.predict(
                source=imagens, 
                save=True, 
                project="output", 
                name="inferencia_visual", 
                conf=0.25
            )
            
            print("\nMétrica de Negócio (Contagem de Parafusos):")
            for i, result in enumerate(resultados):
                # len(result.boxes) retorna a contagem exata de objetos (Bounding Boxes) na imagem
                num_parafusos = len(result.boxes)
                print(f"  -> Imagem [{os.path.basename(imagens[i])}]: {num_parafusos} parafusos contados.")
            
            print(f"\nAs imagens com as caixas desenhadas foram salvas para conferência em: {resultados[0].save_dir}")
        else:
            print(f"Nenhuma imagem encontrada na pasta {pasta_teste}.")
    else:
        print(f"Pasta de testes '{pasta_teste}' não encontrada.")

    print("\nScript de testes finalizado!")

if __name__ == "__main__":
    main()
