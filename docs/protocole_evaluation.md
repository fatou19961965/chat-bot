# Protocole d'évaluation

Objectif : établir, par la mesure, si la chaîne atteint un niveau exploitable.
**Une conclusion négative est un résultat valide** : elle doit simplement
reposer sur des chiffres reproductibles.

## 0. Préalables (dès la phase 0)

1. Licences : lire `licences_et_conformite.md` ; la réponse conditionne toute
   suite commerciale.
2. Conformité : formulaire de consentement, procédure d'anonymisation,
   déclaration CDP **avant** la première collecte.
3. Figer la convention : `guide_transcription.md` validé par l'encadrant.

## 1. Jeu de test

| Partition | Volume | Rôle |
|---|---|---|
| FLEURS wolof (test) | complet | comparabilité avec la littérature (parole lue) |
| Collecte interne | 45–60 min (variante courte : 30 min) | représentativité : spontané, alternance codique, mobile, bruit, deux sexes |

Manifeste CSV (voir `data/test/manifest_example.csv`) :
`audio_path, reference, reference_fr, speaker_id, gender, condition, code_switch, source`.

Règles :
- aucun locuteur du test ne doit figurer dans les données de fine-tuning ;
- `reference_fr` est requis pour la cascade (traduction humaine de la référence) ;
- double annotation d'au moins 20 % → `python -m eval.agreement transcription`.
  Le WER inter-annotateur est le plancher en dessous duquel une différence
  entre systèmes n'est pas interprétable.

## 2. Normalisation (appliquée à la référence ET à l'hypothèse)

NFC, minuscules, balises retirées (`[bruit]`, `<fr>…</fr>` → texte gardé),
ponctuation supprimée, apostrophes internes conservées, hésitations retirées,
diacritiques **conservés**. Le WER sans diacritiques est reporté à part.
Implémentation unique : `app/metrics.normalize_text`, recoupée par jiwer.

## 3. ASR

```bash
python -m eval.asr_benchmark --manifest data/test/manifest.csv --models \
  facebook/mms-1b-all \
  <modele-whisper-wolof-communautaire> \
  openai/whisper-large-v3            # contrôle négatif : son échec fait partie du résultat
```
`facebook/mms-300m` n'a pas de tête CTC : il n'entre au benchmark qu'après
fine-tuning (axe a). Reporter : WER corpus + IC 95 %, CER, WER sans
diacritiques, ventilation par condition / sexe / alternance codique, RTF.

## 4. Traduction

```bash
python -m eval.mt_benchmark --flores-dir data/flores200_dataset --split devtest \
  --models facebook/nllb-200-distilled-600M facebook/nllb-200-1.3B facebook/nllb-200-3.3B
```
Chiffrer le coût du distillé : écart de chrF++ contre écart de latence.
Évaluation humaine : 100 phrases, 2 évaluateurs, adéquation et fluidité 1–5,
kappa pondéré (`eval.agreement humaine`).

## 5. Cascade — résultat central

```bash
python -m eval.cascade --hyps results/asr/facebook__mms-1b-all/hypotheses.csv \
  --threshold <chrF++ jugé acceptable> --synthetic
```
- points réels (sorties ASR effectives) + points synthétiques (bruit contrôlé) ;
- chaque point avec et sans restauration de ponctuation → gain chiffré ;
- `wer_max_tolerable` : WER d'entrée au-delà duquel la traduction passe sous
  le seuil.

Fixer le seuil chrF++ **à partir de l'évaluation humaine** (chrF++ moyen des
phrases jugées adéquates ≥ 4), pas arbitrairement.

## 6. Décision (revue de mi-parcours)

| Situation mesurée | Axe de la phase 3 |
|---|---|
| WER ASR légèrement au-dessus du WER tolérable | (a) fine-tuning de l'adaptateur sur Kallaama + données internes |
| Gain ponctuation significatif | (b) module ponctuation/normalisation appris |
| WER très au-dessus, ou traduction non viable même sur références | (c) classification d'intention fermée (WolBanking77) |

Toute amélioration se mesure **sur le même harnais, le même jeu de test**.

## 7. Suivi en production

Le service journalise tout énoncé ; les corrections alimentent
`/api/stats`, `scripts/calibrate_quality.py` et `scripts/mine_corrections.py`.
La CI rejoue le benchmark ASR sur un commit marqué `[eval]` et échoue en cas
de régression (`scripts/check_regression.py`).
