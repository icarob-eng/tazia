import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# configs
data = os.path.join('data', 'gz2_hart16_classes_simple.csv')
images = os.path.join('data', 'images') 

def main():
    if not os.path.exists(data):
        print(" CSV não encontrado.")
        return

    df = pd.read_csv(data)

    model = VGG16(weights='imagenet', include_top=False, pooling='avg')

    features = []
    ids_salvos = []
    encontradas = 0

    print(f"Procurando imagens na pasta: {os.path.abspath(images)}")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        if 'asset_id' in df.columns:
            nome_arquivo = str(row['asset_id']) + ".jpg"
        else:
            nome_arquivo = str(row['dr7objid']) + ".jpg"

        caminho_completo = os.path.join(images, nome_arquivo)
        
        if os.path.exists(caminho_completo):
            try:
                img = image.load_img(caminho_completo, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                feat = model.predict(x, verbose=0)[0]
                
                features.append(feat)
                ids_salvos.append(str(row['dr7objid']))
                encontradas += 1
            except Exception as e:
                pass
        else:
            if index == 0: 
                print("Verifique se o nome do arquivo na pasta é igual a esse acima!")

    # Salvar
    if encontradas > 0:
        print(f"\n {encontradas} imagens processadas.")
        df_resultado = pd.DataFrame(features)
        df_resultado.insert(0, 'dr7objid', ids_salvos)
        df_resultado.to_pickle(os.path.join('data', 'features_galaxias.pkl'))
        print("Arquivo salvo com sucesso!")
    else:
        print("\nNenhuma imagem encontrada.")

if __name__ == "__main__":
    main()