# Licences et conformité

> Note technique, pas un avis juridique : à faire valider par le juriste.

## Modèles utilisés par défaut

| Composant | Modèle | Licence | Usage commercial |
|---|---|---|---|
| ASR | facebook/mms-1b-all | CC-BY-NC 4.0 | **Non** |
| Traduction (défaut) | Gemma 4 via Ollama (gemma4:*) | Apache 2.0 (à vérifier sur la fiche du modèle) | Oui, a priori |
| Traduction (alternative) | facebook/nllb-200-* | CC-BY-NC 4.0 | **Non** |
| Synthèse | facebook/mms-tts-fra | CC-BY-NC 4.0 | **Non** |
| Contrôle négatif | openai/whisper-large-v3 | MIT | Oui (mais ne couvre pas le wolof) |
| Synthèse alternative | voix du navigateur | selon l'OS | selon l'OS |

Conséquence : la chaîne par défaut convient à l'évaluation, à la recherche et
à la démonstration interne ; **pas à un service facturé**. Un adaptateur
fine-tuné à partir de MMS hérite de la clause NC.

## Pistes pour une exploitation commerciale (à instruire en phase 0)

- ASR : fine-tuning d'un modèle à licence permissive (XLS-R ou w2v-BERT 2.0,
  à vérifier) sur des données dont la licence l'autorise ; modèles Whisper
  fine-tunés communautaires : vérifier la licence du modèle **et** des données
  d'entraînement.
- Traduction : modèles multilingues à licence permissive couvrant le wolof
  (à identifier et mesurer sur le même harnais), ou bascule vers la
  classification d'intention (axe c), entraînée sur des données maîtrisées.
- Synthèse : voix françaises à licence permissive, ou synthèse du navigateur.

Toute substitution se mesure sur le harnais : une licence permissive avec un
WER inexploitable n'est pas une solution.

## Jeux de données

| Jeu | Licence indiquée | Note |
|---|---|---|
| Kallaama | CC-BY 4.0 | domaine agricole ; citer |
| FLEURS | CC-BY 4.0 | parole lue |
| FLORES-200 | CC-BY-SA 4.0 | partage à l'identique pour les dérivés |
| ALFFA, WolBanking77, corpus GalsenAI | à vérifier un par un | noter licence et biais dans le tableau de la phase 0 |

## Données personnelles (Sénégal)

Cadre : loi n° 2008-12 sur la protection des données à caractère personnel et
Commission de Protection des Données Personnelles (CDP) ; vérifier les textes
en vigueur au moment de la collecte.

- Consentement écrit des locuteurs (finalités : évaluation, amélioration des
  modèles ; durée de conservation ; droit de retrait).
- La voix est une donnée personnelle ; selon l'usage, potentiellement
  biométrique.
- Anonymisation des transcriptions (noms, numéros de compte, téléphones) ;
  pseudonymisation des identifiants de locuteurs.
- Déclaration CDP **avant** la collecte.
- En production : `WVC_KEEP_AUDIO=0` tant que le recueil du consentement des
  utilisateurs n'est pas intégré ; purge programmée du répertoire `runtime/`.
