# Routage dynamique de véhicules avec GNN et apprentissage par renforcement

Projet de stage de 4e année d'Ousama Akhrif, consacré au routage dynamique de véhicules (DVRP) dans un environnement incertain, avec fenêtres temporelles et prise en compte du risque.

## Approche

Le notebook combine un encodeur de graphe `TransformerConv` tenant compte des arêtes, un encodeur de véhicule, un décodeur par attention et un biais explicite lié à l'incertitude du trajet. La politique est entraînée par REINFORCE avec un critique appris. Un ordonnanceur sélectionne les véhicules au fil des événements.

## Résultats présentés dans le rapport de stage

La comparaison sur 50 instances de test, avec politiques déterministes, rapporte :

| Méthode | Taux de service | Clients servis en moyenne | Distance totale | km par client servi |
| --- | ---: | ---: | ---: | ---: |
| Plus proche voisin | 0,376 | 56,4 | 218,8 km | 3,88 |
| Heuristique tenant compte de l'urgence | 0,403 | 60,5 | 216,0 km | 3,57 |
| GNN, meilleur point observé | 0,556 | 83,3 | 348,6 km | 4,18 |

Le rapport présente également un test d'intervention sur 200 décisions : le score de décision baisse dans le sens attendu lorsque l'incertitude de l'arête augmente. Dans un second test sur 25 instances, l'incertitude moyenne des arêtes choisies diminue de 14 % lorsque le coefficient de risque passe de 0 à 1. Ces chiffres sont ceux du rapport et ne constituent pas une nouvelle vérification du modèle fourni ici.

## Contenu

- `DVRP_GNN_RL.ipynb` : environnement, modèle, entraînement et évaluation.
- `dvrp_gnn_rl_final.pt` : poids du modèle final.
- `config/` : paramètres du scénario et distribution empirique des demandes.

## Utilisation

Installer les dépendances de `requirements.txt`, ouvrir le notebook depuis la racine du projet, puis suivre les cellules pertinentes. Les sections 1 à 11 définissent les composants. Les cellules d'entraînement des sections 12 et suivantes lancent un entraînement long et créent de nouveaux fichiers ; elles sont facultatives pour consulter l'architecture. La cellule d'évaluation finale pointe vers le modèle fourni à la racine du dossier.

Le chargement d'un checkpoint PyTorch suppose que le fichier provient d'une source fiable. Les versions exactes des bibliothèques utilisées lors de l'entraînement ne sont pas documentées dans les fichiers d'origine.
