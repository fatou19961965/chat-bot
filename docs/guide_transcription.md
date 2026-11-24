# Guide de transcription du wolof

Version 0.1 — à valider par l'encadrant avant toute annotation. Toute
modification ultérieure impose de ré-annoter le sous-ensemble double.

## Orthographe

- **Orthographe officielle CLAD** (alphabet national, décrets sur la
  transcription des langues nationales). Ne pas franciser :
  `jërëjëf`, pas `dieuredieuf`.
- Voyelles longues doublées (`aa`, `ee`, `ii`, `oo`, `uu`, `ëë`, `ée`, `óo`).
- Consonnes géminées doublées (`bb`, `dd`, `gg`, `jj`, `kk`, `ll`, `mm`, `nn`, `ññ`, `pp`, `tt`, `ww`, `yy`).
- `ñ`, `ŋ`, `ë`, `é`, `ó`, `à` toujours écrits : ils distinguent des mots.
- Nombres en toutes lettres (`ñaar`, `fukk`), pas de chiffres.
- Noms propres avec majuscule initiale ; sigles tels que prononcés.

## Ce que l'on transcrit

- Ce qui est dit, pas ce qui aurait dû être dit : ne pas corriger la grammaire.
- Mots tronqués : partie prononcée suivie d'un tiret (`dem-`).
- Hésitations : `euh` (elles sont retirées au calcul du WER, mais utiles
  pour d'autres analyses).
- Répétitions : toutes transcrites.

## Balises

| Balise | Usage |
|---|---|
| `<fr>…</fr>` | segment en français (alternance codique), écrit en orthographe française standard |
| `[bruit]` | bruit non vocal gênant |
| `[rire]` | rire |
| `[inaudible]` | passage incompréhensible (ne pas deviner) |
| `[chevauchement]` | parole superposée d'un autre locuteur, non transcrite |

Exemple : `Dama bëgg ubbi <fr>compte</fr> ci banque bi [bruit].`

Un emprunt intégré au wolof (prononciation adaptée, usage courant) s'écrit
en wolof sans balise ; un mot français prononcé à la française prend `<fr>`.
En cas de doute, noter l'énoncé dans la colonne `commentaire` pour arbitrage.

## Ponctuation

Autorisée pour la lisibilité (elle est retirée au calcul du WER). `?` en fin
de question : utile pour tester l'hypothèse ponctuation.

## Traduction de référence (`reference_fr`)

Français standard, fidèle au sens, sans ajout ni explication ; les segments
`<fr>` sont repris tels quels. Traduire à partir de l'audio **et** de la
transcription.

## Double annotation

Au moins 20 % des énoncés, par deux annotateurs sans concertation. Calcul :
`python -m eval.agreement transcription --a A.csv --b B.csv`. Les énoncés
signalés « à arbitrer » sont revus ensemble ; les décisions prises enrichissent
ce guide.
