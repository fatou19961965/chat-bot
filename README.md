# Làkk — chatbot vocal wolof → français, avec mesure de la qualité

Vous parlez wolof ; l'application transcrit, traduit en français, **lit la
traduction à voix haute**, et affiche pour chaque énoncé **à quel point elle
est fiable**. Chaque correction saisie devient une mesure réelle (WER, CER,
chrF++) et une donnée d'entraînement.

Le projet implémente le sujet de stage *« Chaîne de traitement de la parole
wolof — Évaluation, décision, industrialisation »* : un service durci (phase 4)
**et** le harnais qui permet de décider, chiffres à l'appui, si la chaîne est
exploitable (phases 1–3).

```
 micro ──► VAD + découpage ──► ASR wolof ──► lexique + ponctuation ──► Gemma n-best ──► reclassement ──► TTS français
            (app/audio.py)     (MMS-1B wol)   (app/postprocess.py)     (beam 5, n=3)   aller-retour     (MMS-TTS fra
                                    │                                        │          wo→fr→wo          ou navigateur)
                                    ▼                                        ▼              │
                             confiance CTC                           log-prob moyenne       ▼
                                    └──────────────►  score de qualité estimé  ◄──── chrF aller-retour
                                                             │
                                          journal SQLite (audio, sorties, scores, corrections)
                                                             │
                                          corrections ──► WER / CER / chrF++ réels ──► étalonnage
```

## Démarrage rapide

### Sans GPU ni téléchargement (mode simulé, pour découvrir l'interface)

```bash
pip install -r requirements-dev.txt uvicorn
make run-mock            # http://localhost:8000
```

### Avec les vrais modèles (traduction par Gemma)

La traduction wolof → français est faite par **Gemma** (Google), servi en
local par [Ollama](https://ollama.com). Aucune clé ni compte n'est nécessaire.

```bash
cp .env.example .env     # définir WVC_API_KEYS
docker compose up --build -d
# premier démarrage : ASR + TTS (~4 Go, volume hf-cache) et Gemma (volume ollama)
docker compose logs -f api   # suivre le téléchargement
```

Ou en local :

```bash
# 1. Ollama + Gemma
ollama pull gemma4:e4b          # ou gemma4:e2b (PC modeste), gemma4:12b (GPU)
# 2. Python
pip install torch --index-url https://download.pytorch.org/whl/cpu   # ou CUDA
pip install -r requirements.txt
WVC_PRELOAD=1 WVC_MT_BACKEND=gemma WVC_GEMMA_MODEL=gemma4:e4b uvicorn app.main:app --port 8000
```

Sous Windows : `windows\installer.bat` puis `windows\lancer.bat`
(voir `windows/LISEZMOI.txt`).

### Comment Gemma traduit

| Étape | Détail (`GemmaTranslator`, `app/models.py`) |
|---|---|
| Invite | Consigne de traduction fidèle (wolof parlé, mots français gardés) + les **8 paires les plus proches** de la mémoire de traduction, en exemples (`WVC_GEMMA_EXEMPLES`). Chaque correction enregistrée améliore donc les traductions suivantes. |
| Candidats | Le 1ᵉʳ en décodage glouton (température 0, reproductible), les suivants échantillonnés (`WVC_GEMMA_TEMPERATURE`). |
| Confiance | exp(log-prob moyenne par jeton) renvoyée par Ollama ; à défaut (vieille version d'Ollama), accord entre candidats. |
| Aller-retour | Gemma rétro-traduit le français en wolof ; chrF avec la transcription (`WVC_GEMMA_ROUNDTRIP=0` pour désactiver et diviser le temps par deux). |
| Nettoyage | Les « Traduction : », guillemets, explications ajoutées sont retirés. |

### Choisir le modèle de reconnaissance du wolof

`WVC_ASR_MODEL` accepte tout modèle CTC (MMS, XLS-R, w2v-BERT) ou Whisper ;
le type est détecté automatiquement. Candidats à comparer sur vos propres
enregistrements (WER mesuré par les corrections) :

| Modèle | Type | Atout |
|---|---|---|
| `facebook/mms-1b-all` (défaut) | CTC + adaptateur | référence du projet |
| `CAYTU/whosper-large-v2` | Whisper large | wolof mêlé de français/anglais ; lent sur CPU |
| `M9and2M/whisper_small_wolof_mix_hum_mach_data` | Whisper small | rapide sur CPU |
| `kingabzpro/wav2vec2-large-xlsr-53-wolof` | CTC (XLS-R) | Apache 2.0 |

Revenir à NLLB : `WVC_MT_BACKEND=nllb` (ou `windows\lancer.ps1 -NLLB`).
Comparer les deux sur FLORES : `make bench-gemma`.

Ouvrir `http://localhost:8000`, renseigner la clé d'API dans **Réglages**,
appuyer sur le micro (ou la barre d'espace), parler, appuyer à nouveau.

Pour personnaliser le bandeau avec votre propre photo, déposez-la sous
`app/static/img/hero.jpg` (format large) : elle remplace l'illustration.

Le navigateur n'autorise le micro qu'en `https://` ou sur `localhost` : en
déploiement VPS, placer le service derrière un proxy TLS (Caddy, Nginx).

## Ce qui réduit les erreurs

| Levier | Où | Pourquoi |
|---|---|---|
| Découpage sur les silences (VAD), segments ≤ 20 s | `app/audio.py` | Les modèles CTC saturent au-delà d'une minute ; NLLB traduit mieux phrase par phrase. |
| Normalisation du niveau, filtres navigateur (bruit, écho, gain) | `audio.py`, `index.html` | Enregistrements mobiles faibles ou bruités. |
| Réduction de bruit : passe-haut 80 Hz + soustraction spectrale (`WVC_DEBRUITAGE`) | `audio.denoise` | Ventilateur, clim, moteur, ronflement, foule lointaine atténués avant la reconnaissance. |
| VAD dans la bande de la voix (250–3 800 Hz), rejet des zones à faible rapport signal/bruit (`WVC_SNR_MIN_DB`) et des bruits à spectre plat | `audio.vad_segments` | Les claquements et bruits de fond ne sont plus pris pour de la parole. |
| Segments transcrits avec une confiance < `WVC_ASR_CONF_MIN` ignorés | `pipeline.run` | Le bruit « transcrit » n'est plus traduit ; message « Trop de bruit » à la place. |
| Adaptateur wolof de MMS-1B (et adaptateur fine-tuné en option) | `app/models.py`, `scripts/finetune_mms_adapter.py` | Whisper ne couvre pas le wolof ; MMS oui. |
| Lexique de corrections orthographiques (→ CLAD) | `data/lexique_corrections.tsv` | Variantes récurrentes ; enrichi à partir des corrections (`scripts/mine_corrections.py`). |
| Restauration de ponctuation et casse | `app/postprocess.py` | NLLB est entraîné sur du texte ponctué ; hypothèse mesurée par `eval/cascade.py`. |
| Mémoire de traduction (exacte ou approchée), enrichie par les corrections | `app/memory.py`, `data/memoire_traduction.tsv` | NLLB rend mal les expressions figées (salutations, formules) ; les fautes de frappe proches sont rattrapées. |
| Exemples de la mémoire de traduction dans l'invite de Gemma | `GemmaTranslator._messages` | Gemma apprend en contexte l'orthographe et les formules validées. |
| Plusieurs candidats, reclassement par cohérence aller-retour | `GemmaTranslator.translate`, `Translator.translate` | Écarte les traductions fluides mais infidèles (hallucinations). |
| `no_repeat_ngram_size=4` (NLLB), sortie bornée à 256 jetons (Gemma) | idem | Coupe les boucles de répétition. |
| Alerte « peu fiable » avec conseil ciblé | `app/quality.py` | Mieux vaut demander de répéter que traduire faux. |

## Comment la qualité est suivie

**En direct (sans référence)** — trois signaux par énoncé, combinés en un
score 0–100 % et un verdict *Fiable / À vérifier / Peu fiable* :

- confiance de la reconnaissance (probabilité CTC moyenne hors blancs) ;
- confiance de la traduction (probabilité moyenne par jeton) ;
- cohérence aller-retour : le français est retraduit en wolof et comparé
  (chrF) à la transcription.

Ce score est un **indicateur de risque, pas un WER**. Il est affiché, tracé
sur la session et journalisé.

**Mesuré (avec référence)** — le bouton *Corriger* enregistre la bonne
transcription et/ou la bonne traduction. Le serveur calcule alors WER, CER,
WER sans diacritiques et chrF++, et le panneau *Qualité mesurée* affiche les
moyennes et la corrélation entre score estimé et WER. Avec au moins 20
corrections, `scripts/calibrate_quality.py` réajuste les poids du score.

`GET /api/export.csv` exporte les paires corrigées au format manifeste du
harnais (à faire relire par un annotateur avant tout usage en test).

## Harnais d'évaluation (phases 1 à 3)

Détail et ordre d'exécution : [docs/protocole_evaluation.md](docs/protocole_evaluation.md).

| Étape | Commande | Sortie |
|---|---|---|
| Partition FLEURS | `python -m eval.prepare_fleurs --split test` | manifeste + wav |
| Accord inter-annotateur | `python -m eval.agreement transcription --a A.csv --b B.csv` | WER/CER entre annotateurs, énoncés à arbitrer |
| Benchmark ASR | `python -m eval.asr_benchmark --manifest … --models …` | WER/CER corpus, IC 95 %, ventilation, RTF |
| Benchmark traduction | `python -m eval.mt_benchmark --flores-dir … --models …` | chrF++, spBLEU, deux sens |
| Évaluation humaine | `python -m eval.agreement humaine --init …` puis `--a --b` | moyennes, kappa pondéré |
| **Courbe de dégradation** | `python -m eval.cascade --hyps … --synthetic` | `courbe_degradation.png`, WER maximal tolérable, gain ponctuation |
| Fine-tuning (axe a) | `python scripts/finetune_mms_adapter.py …` | adaptateur à re-mesurer sur le même harnais |

Chaque résultat embarque le commit, les versions des bibliothèques et les
paramètres, pour qu'un tiers retrouve les chiffres.

## API

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/health` | état, modèles chargés, file d'attente |
| POST | `/api/pipeline` | `audio` (multipart), `tts` → transcription, traduction, qualité, WAV français en base64 |
| POST | `/api/translate` | `{"texte": "..."}` → traduction d'un texte wolof |
| POST | `/api/inferences/{id}/correction` | `{"ref_transcription", "ref_traduction"}` → mesures réelles |
| GET | `/api/inferences` | derniers énoncés journalisés |
| GET | `/api/stats` | agrégats estimés et mesurés |
| GET | `/api/export.csv` | paires corrigées |

Toutes les routes `/api/*` sauf `health` exigent `X-API-Key` si
`WVC_API_KEYS` est défini. Sous charge : 429 avec `Retry-After`. Codes
d'erreur : 401 clé, 413 taille, 422 audio illisible/trop long/trop court.

## Tests

```bash
make dev && make test     # modèles simulés : pas de torch, pas de GPU
```

## Arborescence

```
app/        service : audio, modèles, post-traitement, qualité, journal, API, interface
eval/       harnais : FLEURS, ASR, traduction, cascade, accords
scripts/    fine-tuning, étalonnage, minage des corrections, contrôle de régression
docs/       protocole, guide de transcription, licences et conformité, runbook
data/       lexique, manifeste d'exemple (les données réelles ne sont pas versionnées)
tests/      tests unitaires et de bout en bout
```

## Limites connues

- Le score estimé n'est fiable qu'une fois étalonné sur des corrections réelles.
- La VAD énergétique est moins robuste au bruit qu'un modèle appris ;
  le module est remplaçable.
- L'alternance codique wolof/français n'est pas gérée par MMS (modèle
  monolingue) : elle est détectée, pénalisée dans le score, et à quantifier
  dans le jeu de test.
- **Licences** : MMS, NLLB-200 et MMS-TTS sont en CC-BY-NC 4.0. Le service tel
  quel ne peut pas être exploité commercialement. Voir
  [docs/licences_et_conformite.md](docs/licences_et_conformite.md).
