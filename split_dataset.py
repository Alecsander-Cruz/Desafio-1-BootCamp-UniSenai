import os
import shutil
import random

def split_dataset(src_dir="dataset", dest_dir="dataset_split", train_ratio=0.8):
    """
    Divide o dataset original em treino (80%) e validação (20%).
    A pasta 'test' é criada vazia apenas para manter a estrutura do YOLO.
    """
    src_images = os.path.join(src_dir, "images")
    src_labels = os.path.join(src_dir, "labels")

    if not os.path.exists(src_images):
        print(f"Erro: Pasta '{src_images}' não encontrada.")
        return
    if not os.path.exists(src_labels):
        print(f"Erro: Pasta '{src_labels}' não encontrada.")
        return

    # 1. Criar estrutura de pastas do destino
    print("Criando estrutura de pastas...")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, split, "labels"), exist_ok=True)

    # 2. Pegar a lista de imagens e embaralhar
    images = [f for f in os.listdir(src_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.seed(42) # Seed fixa garante que se você rodar 2x, o split será idêntico (não mistura as fotos)
    random.shuffle(images)

    # 3. Calcular quantidade de arquivos
    total_images = len(images)
    train_size = int(total_images * train_ratio)
    
    train_images = images[:train_size]
    val_images = images[train_size:]

    # Função auxiliar para copiar os pares de imagem/label
    def copy_files(file_list, split_name):
        print(f"Copiando {len(file_list)} arquivos para a pasta '{split_name}'...")
        for img_name in file_list:
            # Caminhos de origem
            img_src = os.path.join(src_images, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_src = os.path.join(src_labels, label_name)

            # Caminhos de destino
            img_dest = os.path.join(dest_dir, split_name, "images", img_name)
            label_dest = os.path.join(dest_dir, split_name, "labels", label_name)

            # Copiar Imagem
            shutil.copy(img_src, img_dest)
            
            # Copiar Label (se existir)
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dest)

    print(f"\nIniciando Split do Dataset (Total: {total_images} imagens)")
    print(f"   -> Treino: {train_ratio*100:.0f}% | Validação: {(1-train_ratio)*100:.0f}%")
    
    # 4. Executar a cópia
    copy_files(train_images, "train")
    copy_files(val_images, "val")
    
    # OBS: O test_mobile.py agora puxa imagens de 'dataset_split/test/images'.
    # Para o script não falhar ao rodar, vou copiar umas 3 imagens do val para lá como amostra crua.
    if len(val_images) > 0:
        amostra_teste = val_images[:min(3, len(val_images))]
        copy_files(amostra_teste, "test")
        print("Nota: Copiei algumas imagens do 'val' para o 'test' para você poder rodar seu test_mobile.py.")
    
    print("\nDataset dividido com sucesso e pronto para o YOLO na pasta 'dataset_split'!")

if __name__ == "__main__":
    # Limpa a pasta dataset_split inteira antes se ela já existir, 
    # para evitar arquivos "fantasmas" de execuções anteriores misturando os splits.
    dest_path = "dataset_split"
    if os.path.exists(dest_path):
        print(f"Limpando pasta '{dest_path}' antiga para evitar conflitos...")
        shutil.rmtree(dest_path)
        
    split_dataset(train_ratio=0.8)
