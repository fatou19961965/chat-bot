# Harnais d'evaluation complet (phases 1 et 2 du stage)
param(
    [int]$Limite = 50,          # 0 = jeu complet (plusieurs heures sur CPU)
    [switch]$SansWhisper,       # saute le controle negatif Whisper (3 Go, tres lent)
    [string]$Gemma = "gemma4:e4b"  # modele Gemma compare a NLLB
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Environnement .venv introuvable dans $(Get-Location). Lancez ce script depuis le dossier ou .venv a ete cree." -ForegroundColor Red
    exit 1
}
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue

Write-Host "== 1. Dependances d'evaluation ==" -ForegroundColor Cyan
& $py -m pip install -r requirements-eval.txt

Write-Host "== 2. FLEURS wolof (audio de test) ==" -ForegroundColor Cyan
if (-not (Test-Path "data\fleurs\manifest_test.csv")) { & $py -m eval.prepare_fleurs --split test }

Write-Host "== 3. FLORES (textes wolof / francais) ==" -ForegroundColor Cyan
if (-not (Test-Path "data\flores200_dataset\devtest\fra_Latn.devtest")) {
    Write-Host "Acces restreint : compte huggingface.co + conditions acceptees sur openlanguagedata/flores_plus" -ForegroundColor Yellow
    hf auth login
    & $py -m eval.export_flores --out data/flores200_dataset
}

Write-Host "== 4. References francaises pour FLEURS ==" -ForegroundColor Cyan
& $py -m eval.add_flores_fr --manifest data/fleurs/manifest_test.csv --flores-dir data/flores200_dataset --out data/fleurs/manifest_test_fr.csv

$lim = @(); if ($Limite -gt 0) { $lim = @("--limit", "$Limite") }

Write-Host "== 5. Benchmark transcription ==" -ForegroundColor Cyan
$modeles = @("facebook/mms-1b-all"); if (-not $SansWhisper) { $modeles += "openai/whisper-large-v3" }
& $py -m eval.asr_benchmark --manifest data/fleurs/manifest_test_fr.csv --models $modeles @lim

Write-Host "== 6. Benchmark traduction ==" -ForegroundColor Cyan
$limMt = @(); if ($Limite -gt 0) { $limMt = @("--limit", "$([Math]::Max($Limite, 200))") }
$env:WVC_OLLAMA_URL = "http://127.0.0.1:11434"
& $py -m eval.mt_benchmark --flores-dir data/flores200_dataset --models $Gemma facebook/nllb-200-distilled-600M @limMt

Write-Host "== 7. Courbe de degradation en cascade ==" -ForegroundColor Cyan
& $py -m eval.cascade --hyps results/asr/facebook__mms-1b-all/hypotheses.csv --synthetic --mt-model $Gemma

Write-Host ""
Write-Host "Resultats :" -ForegroundColor Green
Get-Content results\asr\summary.md
Get-Content results\mt\summary.md
Get-Content results\cascade\summary.json
if (Test-Path results\cascade\courbe_degradation.png) { Start-Process results\cascade\courbe_degradation.png }
