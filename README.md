# Routage dynamique de véhicules avec GNN et apprentissage par renforcement

Ce projet de stage étudie la planification de tournées lorsque les demandes et les conditions de circulation changent au cours de la journée. À chaque décision, un véhicule disponible doit choisir sa prochaine destination en tenant compte des commandes encore actives, des capacités, des fenêtres de temps et de l'incertitude sur les trajets. Le projet associe un simulateur de DVRP, une politique fondée sur un graphe et un entraînement par renforcement.

## Architecture

```text
Scénario et demandes → simulateur dynamique → observation du réseau
                                                  │
                             véhicule disponible ←┘
                                      │
                  graphe et état véhicule → GNN + encodeur véhicule
                                      │
                         attention + score de risque
                                      │
                     masque de faisabilité → prochaine destination
                                      │
                       nouvel état + récompense → apprentissage RL
```

La décision est **événementielle** : le simulateur avance jusqu'au prochain véhicule disponible. Un ordonnanceur choisit ce véhicule selon sa disponibilité ; ce choix n'est pas appris. La politique neuronale choisit ensuite le client à visiter ou le retour au dépôt. Cette séparation permet d'appliquer les mêmes paramètres de politique à toute la flotte, tout en tenant compte de l'état particulier de chaque véhicule.

### 1. Simulateur et contraintes

L'environnement construit des journées à partir des paramètres de [`config/scenario_params.json`](config/scenario_params.json) et d'un réservoir empirique de demandes [`config/demand_pool_solomon.csv`](config/demand_pool_solomon.csv). Il initialise le dépôt, les clients et la flotte, puis fait évoluer le scénario : nouvelles demandes, annulations, périodes de trafic et variations aléatoires des temps de trajet. Chaque action met à jour la position et l'heure du véhicule, sa capacité restante et le statut des commandes.

Avant la décision du modèle, un **masque d'actions** écarte les destinations non admissibles selon l'état courant : client indisponible ou déjà traité, demande dépassant la capacité, fenêtre de temps ou durée de service incompatible, fin de journée, ou retour au dépôt. Le masque repose sur l'information disponible au moment de décider ; le temps de trajet réalisé peut ensuite différer de l'estimation à cause de l'aléa simulé.

### 2. Représentation de l'état

L'observation réunit trois types d'information : les caractéristiques des nœuds (dépôt et clients), celles des véhicules et celles des arêtes. Le graphe relie tous les nœuds ; ses arêtes portent notamment la distance, le temps de trajet de référence, le temps estimé dans le trafic courant et l'incertitude. Les attributs d'arête sont **propres au véhicule considéré** : deux véhicules ne voient pas nécessairement le même coût de déplacement au même instant.

Les observations sont converties en objets PyTorch Geometric. La topologie des arêtes est réutilisée d'un état à l'autre, tandis que les attributs et les statuts dynamiques sont actualisés. Le graphe sert à donner au modèle une vue des relations entre destinations, au-delà d'un simple classement isolé des clients.

### 3. Politique GNN et sélection d'une destination

L'encodeur des nœuds applique une couche **TransformerConv** sensible aux attributs d'arête. Une porte apprise module ces attributs avant le passage dans la couche ; une connexion résiduelle préserve les informations initiales. Un contexte agrégé par zone apporte également une information spatiale plus large. Pendant l'entraînement, un DropEdge régularise l'encodeur ; l'évaluation utilise le graphe complet. En parallèle, un petit réseau encode l'état du véhicule sélectionné.

Le décodeur combine la représentation du véhicule et un résumé global du graphe. Une attention multi-tête (*glimpse*) construit le contexte de décision, puis une attention de type *pointer* attribue un score aux destinations. Deux éléments interviennent avant le choix final : le masque de faisabilité supprime les actions interdites et un terme de risque réduit le score des trajets plus incertains. À l'entraînement, l'action est échantillonnée ; lors de l'évaluation déterministe, le meilleur score est choisi.

Ce mécanisme distingue donc trois rôles : **le simulateur** définit la situation et les contraintes, **l'ordonnanceur** désigne le véhicule disponible, et **la politique apprise** choisit sa prochaine destination.

### 4. Récompense et apprentissage

Après chaque déplacement, la récompense valorise une livraison accomplie et pénalise le temps de trajet, le retard et l'exposition au risque. À la fin de l'épisode, des pénalités tiennent compte notamment des demandes non servies et des dépassements de service. L'optimisation vise ainsi un compromis entre service, délais et coût de déplacement, plutôt que la distance seule.

La politique est entraînée avec **REINFORCE** sur les retours de l'épisode. Un critique appris estime une valeur de référence pour réduire la variance de l'apprentissage ; politique et critique disposent de leurs propres optimiseurs. Le notebook prévoit la normalisation des retours, le contrôle des gradients et la reprise de l'entraînement depuis un checkpoint. Le fichier [`dvrp_gnn_rl_final.pt`](dvrp_gnn_rl_final.pt) contient les poids fournis pour l'évaluation.

## Évaluation

La politique est comparée à deux règles de décision : **plus proche voisin**, qui privilégie la distance, et une **heuristique d'urgence**, qui prend en compte les échéances. Les méthodes sont exécutées sur les mêmes graines de test. Le notebook suit le taux de service ajusté pour les annulations, les clients servis, la distance, le temps, les retards, les rejets et les dépassements. Il contient aussi des diagnostics sur les actions choisies, les probabilités attribuées aux destinations et les composantes de la récompense.

Les résultats suivants, issus du rapport de stage, portent sur **50 instances de test** avec des politiques déterministes :

| Politique | Taux de service | Clients servis en moyenne | Distance totale moyenne |
| --- | ---: | ---: | ---: |
| Plus proche voisin | 37,6 % | 56,4 | 218,8 km |
| Urgence | 40,3 % | 60,5 | 216,0 km |
| GNN, meilleur point observé | 55,6 % | 83,3 | 348,6 km |

La politique GNN sert davantage de clients dans cette comparaison, mais parcourt aussi une distance plus grande. Le tableau ne démontre donc pas une amélioration sur tous les critères. Ces chiffres proviennent du rapport de stage : les sorties détaillées de cette campagne ne sont pas incluses dans l'archive et n'ont pas été recalculées pour ce dépôt.

## Organisation du dépôt

- [`DVRP_GNN_RL.ipynb`](DVRP_GNN_RL.ipynb) : simulateur, conversion en graphes, modèle, entraînement, diagnostics et comparaison.
- [`config/scenario_params.json`](config/scenario_params.json) : paramètres du scénario dynamique.
- [`config/demand_pool_solomon.csv`](config/demand_pool_solomon.csv) : réservoir de demandes servant à générer les instances.
- [`dvrp_gnn_rl_final.pt`](dvrp_gnn_rl_final.pt) : checkpoint du modèle.
- [`requirements.txt`](requirements.txt) : dépendances Python principales.

## Exécution

Depuis la racine du dépôt :

```bash
python -m venv .venv
# Windows PowerShell : .\.venv\Scripts\Activate.ps1
# macOS/Linux : source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install notebook
jupyter notebook DVRP_GNN_RL.ipynb
```

Les premières sections du notebook définissent l'environnement et la politique. L'entraînement est coûteux et peut être ignoré pour étudier l'architecture ; la section finale charge le checkpoint fourni et lance la comparaison aux heuristiques. Les sorties de cellules ne sont pas conservées dans le notebook distribué. Les versions exactes des bibliothèques utilisées lors de l'entraînement initial ne sont pas documentées, si bien qu'une nouvelle exécution peut produire des résultats différents.
