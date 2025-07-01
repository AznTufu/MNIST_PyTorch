# 🎨 Reconnaissance de Chiffres Manuscrits avec ONNX

Ce projet utilise un modèle CNN (Convolutional Neural Network) entraîné sur le dataset MNIST pour reconnaître des chiffres manuscrits dessinés par l'utilisateur dans le navigateur.

## 🚀 Fonctionnalités

- **Interface de dessin interactive** : Canvas HTML5 pour dessiner les chiffres
- **Prédiction en temps réel** : Utilisation d'ONNX Runtime Web pour l'inférence
- **Visualisation des probabilités** : Barres de probabilité pour chaque chiffre (0-9)
- **Interface responsive** : Compatible desktop et mobile
- **Prétraitement automatique** : Redimensionnement et normalisation comme lors de l'entraînement

## 📁 Structure du projet

```
htmljs/
├── index.html          # Interface utilisateur principale
├── style.css           # Styles et animations
├── app.js              # Logique JavaScript et intégration ONNX
├── cnn_model.onnx      # Modèle CNN exporté depuis PyTorch
└── model.ipynb         # Modèle PyTorch
```

## 🔧 Technologies utilisées

- **HTML5 Canvas** : Pour l'interface de dessin
- **CSS3** : Styles modernes avec animations et design responsive
- **JavaScript ES6+** : Logique applicative
- **ONNX Runtime Web** : Exécution du modèle de machine learning
- **CDN ONNX Runtime** : `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js`

## 🧠 Caractéristiques du modèle

Le modèle CNN utilisé a les spécifications suivantes :

- **Architecture** :
  - 3 blocs convolutionnels avec BatchNorm et Dropout
  - Conv1: 1→32 canaux
  - Conv2: 32→64 canaux  
  - Conv3: 64→128 canaux
  - Couches fully connected : 128×4×4 → 512 → 256 → 10

- **Prétraitement** :
  - Redimensionnement : 28×28 pixels
  - Normalisation : [0,1] → [-1,1] avec mean=0.5, std=0.5
  - Inversion des couleurs (fond blanc → noir)

- **Augmentation de données (entraînement)** :
  - Rotation aléatoire (±10°)
  - Translation aléatoire (±10%)
  - Perspective aléatoire

## 🌐 Utilisation

1. **Ouvrir le projet** :
   ```bash
   # Servir les fichiers via un serveur web local
   # Par exemple avec Python :
   python -m http.server 8000
   ```

2. **Accéder à l'application** :
   - Ouvrir `http://localhost:8000` dans le navigateur

3. **Utiliser l'interface** :
   - Dessiner un chiffre (0-9) dans la zone de canvas
   - Cliquer sur "Prédire" pour obtenir le résultat
   - Voir les probabilités pour chaque chiffre
   - Utiliser "Effacer" pour recommencer

## 🎯 Précision du modèle

Le modèle a été entraîné avec :
- **Dataset** : MNIST (60,000 images d'entraînement, 10,000 de test)
- **Optimiseur** : AdamW avec learning rate 1e-3
- **Régularisation** : Dropout (0.3) et BatchNormalization
- **Epochs** : 5 (peut être ajusté selon les besoins)

## 📈 Résultat

```bash
Test Error: 
 Accuracy: 98.4%, Avg loss: 0.047277 

loss: 0.173773  [   64/60000]
loss: 0.041781  [ 6464/60000]
loss: 0.159822  [12864/60000]
loss: 0.133094  [19264/60000]
loss: 0.126833  [25664/60000]
loss: 0.022532  [32064/60000]
loss: 0.032363  [38464/60000]
loss: 0.134337  [44864/60000]

Test Error: 
 Accuracy: 99.2%, Avg loss: 0.026052 
```
