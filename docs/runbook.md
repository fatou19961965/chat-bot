# Runbook d'exploitation

## Démarrer / arrêter
```bash
docker compose up -d --build
docker compose logs -f api
docker compose down
```
Premier démarrage : téléchargement des poids (~6 Go) ; le healthcheck laisse
5 minutes. `WVC_PRELOAD=1` charge les modèles au démarrage plutôt qu'à la
première requête.

## Ressources indicatives
- CPU seul : fonctionne, mais plusieurs secondes par énoncé (NLLB avec
  reclassement ≈ 3 générations par segment). Réduire `WVC_N_BEST=1` pour
  diviser la latence de traduction, au prix du reclassement.
- GPU 16 Go : confortable pour MMS-1B + NLLB-600M + MMS-TTS.
- `WVC_MAX_CONCURRENT=1` sur CPU ; augmenter avec prudence sur GPU.

## Symptômes fréquents
| Symptôme | Cause probable | Action |
|---|---|---|
| 429 fréquents | file saturée | augmenter `WVC_MAX_QUEUE` ou les ressources ; vérifier la latence |
| 422 « Audio illisible » | format non décodé | vérifier ffmpeg dans l'image ; tester le fichier avec `ffmpeg -i` |
| Micro inaccessible dans le navigateur | page servie en http | proxy TLS devant le service |
| Scores « peu fiable » généralisés | micro, bruit, ou alternance codique | consulter `/api/inferences` : confiances ASR vs MT |
| Mémoire saturée | modèles 1.3B/3.3B sur CPU | revenir au distillé 600M |

## Sauvegarde
`runtime/inferences.sqlite3` et `runtime/audio/` constituent l'actif de
données : sauvegarde quotidienne chiffrée, rétention conforme au consentement.

## Changer de modèle
1. Mesurer le candidat sur le harnais (`eval.asr_benchmark`).
2. Commit avec `[eval]` : la CI compare à la référence.
3. Mettre à jour `WVC_ASR_MODEL` / `WVC_MT_MODEL`, redémarrer.

## Gemma / Ollama

- Vérifier : `curl http://127.0.0.1:11434/api/tags` doit lister le modèle
  (`WVC_GEMMA_MODEL`, par défaut `gemma4:e4b`).
- « Ollama injoignable » au démarrage : lancer l'application Ollama ou
  `ollama serve` ; en Docker, `docker compose logs ollama`.
- Trop lent sur CPU : `WVC_GEMMA_MODEL=gemma4:e2b`, `WVC_N_BEST=1`,
  `WVC_GEMMA_ROUNDTRIP=0` (2 appels → 1 appel par segment).
- Confiance toujours ≈ 60 % : Ollama trop ancien pour renvoyer les
  logprobs (≥ 0.12.11 requis) → mettre Ollama à jour.

