# 📦 Guide d'Installation - Analyse Campagne Marketing

## Installation de Python

### Windows
1. Téléchargez Python 3.11 depuis [python.org](https://www.python.org/downloads/)
2. Exécutez l'installateur
3. ⚠️ **IMPORTANT**: Cochez "Add Python to PATH" pendant l'installation
4. Vérifiez l'installation :
   ```bash
   python --version
   ```

### macOS
```bash
# Via Homebrew (recommandé)
brew install python@3.11

# Vérification
python3 --version
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3-pip python3-venv

# Vérification
python3 --version
```

---

## 📚 Installation des Bibliothèques

### Option 1 : Installation Complète avec Environnement Virtuel (Recommandé)

#### Étape 1 : Créer un environnement virtuel
```bash
# Naviguer vers le dossier du projet
cd /chemin/vers/rush_1

# Créer l'environnement virtuel
python -m venv .venv
```

#### Étape 2 : Activer l'environnement virtuel
```bash
# Windows (CMD)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

#### Étape 3 : Installer les dépendances
```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer toutes les bibliothèques depuis requirements.txt
pip install -r requirements.txt
```

#### Étape 4 : Lancer Jupyter Notebook
```bash
jupyter notebook
```

---

### Option 2 : Installation Rapide (Sans environnement virtuel)

```bash
# Installation directe de toutes les bibliothèques
pip install pandas numpy matplotlib seaborn plotly scikit-learn scipy xgboost jupyter notebook ipykernel
```

---

## 🔍 Vérification de l'Installation

### Via Terminal
```bash
# Vérifier les packages installés
pip list

# Vérifier une bibliothèque spécifique
python -c "import pandas; print(pandas.__version__)"
```

### Via Jupyter Notebook
Ouvrez le notebook `Analyse Campagne Marketing.ipynb` et exécutez la première cellule de vérification.

---

## 🛠️ Résolution des Problèmes Courants

### Problème : `pip` n'est pas reconnu
**Solution Windows:**
```bash
python -m pip install --upgrade pip
```

**Solution macOS/Linux:**
```bash
python3 -m pip install --upgrade pip
```

### Problème : Erreur d'installation de XGBoost
**Solution:**
```bash
# Windows: Installer Visual C++ Build Tools d'abord
# Puis réessayer:
pip install xgboost

# macOS:
brew install libomp
pip install xgboost
```

### Problème : Jupyter ne démarre pas
**Solution:**
```bash
pip install --upgrade jupyter notebook ipykernel
python -m ipykernel install --user
```

### Problème : Erreur de permissions (macOS/Linux)
**Solution:**
```bash
pip install --user -r requirements.txt
```

---

## 📋 Liste des Bibliothèques

| Bibliothèque | Version | Utilisation |
|--------------|---------|-------------|
| pandas | ≥2.0.0 | Manipulation de données |
| numpy | ≥1.24.0 | Calculs numériques |
| matplotlib | ≥3.7.0 | Visualisations statiques |
| seaborn | ≥0.12.0 | Visualisations statistiques |
| plotly | ≥5.14.0 | Visualisations interactives |
| scikit-learn | ≥1.3.0 | Machine Learning |
| scipy | ≥1.10.0 | Calculs scientifiques |
| xgboost | ≥2.0.0 | Gradient Boosting |
| jupyter | ≥1.0.0 | Interface notebook |

---

## 🚀 Commandes Rapides

```bash
# Tout en une fois (pour les pressés)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && jupyter notebook

# Windows (PowerShell)
python -m venv .venv; .venv\Scripts\Activate.ps1; pip install -r requirements.txt; jupyter notebook
```

---

## 💡 Conseils

- ✅ Utilisez toujours un environnement virtuel pour isoler les dépendances
- ✅ Mettez à jour pip régulièrement : `pip install --upgrade pip`
- ✅ Vérifiez la compatibilité de votre version Python (≥3.8)
- ✅ Gardez vos bibliothèques à jour : `pip install --upgrade -r requirements.txt`

---

## 📞 Support

En cas de problème persistant :
1. Vérifiez que Python est correctement installé : `python --version`
2. Vérifiez que pip fonctionne : `pip --version`
3. Essayez de réinstaller une bibliothèque spécifique : `pip install --force-reinstall nom_bibliotheque`
4. Consultez la documentation officielle de chaque bibliothèque

---

**✨ Une fois l'installation terminée, ouvrez le notebook et commencez l'analyse !**
