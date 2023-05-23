#importando librerias
import cv2
import numpy as np
from google.colab import drive
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# para utilizar google drive (de donde saque las imagenes)
drive.mount('/content/drive')

#carganddo imagenes a detectar y las anotaciones
img_folder = '/content/drive/MyDrive/hands600'
seg_folder = '/content/drive/MyDrive/skins_hand600'
print
img_names = ['/Hand_0000038.jpg', '/Hand_0000101.jpg', '/Hand_0000259.jpg', '/Hand_0000411.jpg']
imgs = [cv2.imread(img_folder + img_name) for img_name in img_names]
segs = [cv2.imread(seg_folder + img_name, cv2.IMREAD_GRAYSCALE) for img_name in img_names]

# definiendo el rango de color de piel en el espacio de color YCrCb
lower_skin = np.array([0, 135, 85], dtype=np.uint8)
upper_skin = np.array([255, 180, 135], dtype=np.uint8)
print
#deteccion piel
preds = []
for i, img in enumerate(imgs):
    # convertir  imagen a espacio de color YCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    #mascara para detectar piel
    skin_mask = cv2.inRange(img_ycrcb, lower_skin, upper_skin)
    skin = cv2.bitwise_and(img, img, mask=skin_mask)
    
    # guardando la predicción
    preds.append(skin_mask)
    
    # Visualizando la imagen original, la segmentación de la pagina y la segmentación automática
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Imagen original')
    axs[1].imshow(segs[i], cmap='gray')
    axs[1].set_title('Segmentación de pagina 11kHand')
    axs[2].imshow(skin_mask, cmap='gray')
    axs[2].set_title('Segmentación automática')
    plt.show()

#convirtiendo las segmentaciones de la pagina y predicciones a vectores
y_true = np.concatenate([seg.flatten() for seg in segs])
y_pred = np.concatenate([pred.flatten() for pred in preds])
#caálculo de medidas de desempeño
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
roc_auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
#print de desempenos
print('Precision: ', precision)
print('Recall: ', recall)
print('F1-score: ', f1)
print('ROC-AUC: ', roc_auc)
