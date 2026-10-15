# Routage dynamique de véhicules avec GNN et apprentissage par renforcement

Projet de stage consacré à la planification de tournées dans un environnement où les demandes évoluent pendant la journée. Les véhicules doivent servir des clients soumis à des fenêtres de temps, tandis que de nouvelles demandes, des annulations et l’incertitude des trajets modifient les décisions à prendre.

L’objectif est d’apprendre une politique de décision capable de tenir compte à la fois de la structure du réseau, de l’état des véhicules et du risque associé aux déplacements.

## Approche

Le projet repose sur un environnement de simulation paramétrable, un **encodeur de graphe TransformerConv** qui représente les clients et les trajets, puis un mécanisme d’attention qui choisit la prochaine action d’un véhicule. Un terme explicite de risque influe sur la décision. La politique est entraînée par **REINFORCE avec critique appris** et comparée à deux heuristiques : plus proche voisin et priorité tenant compte de l’urgence.

```text
Demandes et trafic → état du réseau → GNN → choix du véhicule et du client
                                          ↓
                              récompense et entraînement RL
```

## Résultats

Les résultats suivants, issus du rapport de stage, comparent des politiques déterministes sur **50 instances de test** :

| Politique | Taux de service | Clients servis en moyenne | Distance totale moyenne |
| --- | ---: | ---: | ---: |
| Plus proche voisin | 37,6 % | 56,4 | 218,8 km |
| Urgence | 40,3 % | 60,5 | 216,0 km |
| GNN, meilleur point observé | 55,6 % | 83,3 | 348,6 km |

La politique GNN sert davantage de clients dans cette comparaison, au prix d’une distance totale plus élevée. Ces chiffres proviennent du rapport de stage : les sorties détaillées de cette campagne ne sont pas incluses dans l’archive et n’ont pas été recalculées lors de sa préparation.

## Contenu du projet

- [`DVRP_GNN_RL.ipynb`](DVRP_GNN_RL.ipynb) : environnement, architecture GNN, entraînement, diagnostics et comparaison aux heuristiques.
- `dvrp_gnn_rl_final.pt` : poids du modèle entraîné utilisés par la cellule d’évaluation finale.
- `config/` : paramètres du scénario et distribution empirique des demandes.
- `requirements.txt` : dépendances Python principales.

## Exécution

Créer un environnement Python, l’activer, installer les dépendances, puis ouvrir le notebook depuis la racine du projet :

```bash
python -m venv .venv
# Windows PowerShell : .\.venv\Scripts\Activate.ps1
# macOS/Linux : source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install notebook
jupyter notebook DVRP_GNN_RL.ipynb
```

Dans le notebook, les premières sections définissent l’environnement, le modèle et les fonctions nécessaires. La section d’entraînement lance un calcul long et peut être ignorée pour consulter l’architecture ou utiliser le checkpoint fourni. La section finale charge ce checkpoint et compare la politique aux heuristiques sur les graines de test définies dans le notebook.

Le notebook n’est pas livré avec des sorties de cellules exécutées. Les versions exactes des bibliothèques employées pendant l’entraînement d’origine ne sont pas documentées dans les fichiers disponibles ; les résultats peuvent donc varier lors d’une nouvelle exécution.
