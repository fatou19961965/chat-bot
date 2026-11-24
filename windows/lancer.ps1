# Lance le chatbot vocal (traduction par Gemma via Ollama, par defaut)
param(
    [switch]$Simulation,                 # interface seule, sans modeles
    [switch]$Rapide,                     # traduction tres rapide (NLLB int8, CTranslate2)
    [switch]$NLLB,                       # ancien moteur de traduction (NLLB 600M)
    [string]$Gemma = "gemma4:e4b",       # gemma4:e2b (plus leger) | gemma4:e4b | gemma4:12b | gemma3:4b
    [switch]$Assistant,                  # reponse de l'assistant en francais (meme Gemma)
    [switch]$Bruyant,
    [switch]$SansDebruitage,             # aucune reduction de bruit (pour comparer)
    [switch]$Precis,                     # verification aller-retour (traduction 2x plus lente)
    # Reconnaissance vocale : mms (defaut) | whosper (wolof + francais melanges, lent)
    #                        | whisper-small (rapide) | xlsr | <identifiant Hugging Face>
    [string]$ASR = "mms",                    # lieu tres bruyant : reduction de bruit plus forte
    [int]$Port = 8000
)
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Environnement .venv introuvable dans $(Get-Location). Lancez ce script depuis le dossier ou .venv a ete cree." -ForegroundColor Red
    exit 1
}

$env:WVC_PRELOAD   = "1"
$env:WVC_N_BEST    = "1"
$env:WVC_NUM_BEAMS = "2"
if ($Simulation) { $env:WVC_MOCK_MODELS = "1" } else { $env:WVC_MOCK_MODELS = "0" }
if ($Precis) { $env:WVC_GEMMA_ROUNDTRIP = "1" } else { $env:WVC_GEMMA_ROUNDTRIP = "0" }
# -Bruyant : debruitage plus fort et petits bruits isoles ignores, mais seuil
# de signal/bruit inchange pour ne pas rejeter une vraie phrase dite doucement.
if ($Bruyant) { $env:WVC_DEBRUITAGE = "1.8"; $env:WVC_SNR_MIN_DB = "10"; $env:WVC_VAD_MIN_SILENCE_S = "0.5"; $env:WVC_VAD_MIN_PAROLE_S = "0.25" } else { $env:WVC_DEBRUITAGE = "1.0"; $env:WVC_SNR_MIN_DB = "10"; Remove-Item Env:WVC_VAD_MIN_SILENCE_S -ErrorAction SilentlyContinue; Remove-Item Env:WVC_VAD_MIN_PAROLE_S -ErrorAction SilentlyContinue }
if ($SansDebruitage) { $env:WVC_DEBRUITAGE = "0" }
if ($Assistant) { $env:WVC_LLM_ENABLED = "1" } else { $env:WVC_LLM_ENABLED = "0" }

if ($Rapide) {
    $env:WVC_MT_BACKEND = "rapide"
} elseif ($NLLB) {
    $env:WVC_MT_BACKEND = "nllb"
} else {
    $env:WVC_MT_BACKEND   = "gemma"
    $env:WVC_GEMMA_MODEL  = $Gemma
    $env:WVC_OLLAMA_URL   = "http://127.0.0.1:11434"
    if (-not $Simulation) {
        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            Write-Host "Ollama introuvable : lancez d'abord windows\installer.bat" -ForegroundColor Red; exit 1
        }
        # Demarre le serveur Ollama s'il ne repond pas deja
        try { Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null }
        catch {
            Write-Host "Demarrage d'Ollama..." -ForegroundColor Cyan
            Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
            Start-Sleep -Seconds 5
        }
        ollama show $Gemma *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Telechargement de $Gemma (plusieurs Go, une seule fois)..." -ForegroundColor Yellow
            ollama pull $Gemma
        }
    }
}

$asrModeles = @{
    "mms"           = "facebook/mms-1b-all"
    "whosper"       = "CAYTU/whosper-large-v2"
    "whisper-small" = "M9and2M/whisper_small_wolof_mix_hum_mach_data"
    "xlsr"          = "kingabzpro/wav2vec2-large-xlsr-53-wolof"
}
if ($asrModeles.ContainsKey($ASR)) { $env:WVC_ASR_MODEL = $asrModeles[$ASR] } else { $env:WVC_ASR_MODEL = $ASR }

# hors ligne seulement si les modeles choisis (ASR, TTS) sont deja telecharges
$hub = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$asrCache = Join-Path $hub ("models--" + $env:WVC_ASR_MODEL.Replace("/", "--"))
$ttsCache = Join-Path $hub "models--facebook--mms-tts-fra"
# (seulement pour MMS : les autres modeles peuvent dependre d'un modele de base a telecharger)
if (($ASR -eq "mms") -and (Test-Path $asrCache) -and (Test-Path $ttsCache) -and -not $NLLB -and -not $Rapide) { $env:HF_HUB_OFFLINE = "1" } else { Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue }

# Ouvre le navigateur des que le serveur repond
Start-Job -ArgumentList $Port -ScriptBlock {
    param($p)
    for ($i = 0; $i -lt 180; $i++) {
        try { Invoke-RestMethod "http://127.0.0.1:$p/api/health" -TimeoutSec 2 | Out-Null; Start-Process "http://127.0.0.1:$p"; break }
        catch { Start-Sleep -Seconds 2 }
    }
} | Out-Null

Write-Host "Chargement des modeles (1 a 3 minutes)... le navigateur s'ouvrira tout seul." -ForegroundColor Cyan
Write-Host "Reconnaissance : $env:WVC_ASR_MODEL" -ForegroundColor Cyan
Write-Host "Traduction : $(if ($Rapide) { 'rapide (NLLB int8)' } elseif ($NLLB) { 'NLLB' } else { $Gemma })    Pour arreter : Ctrl+C" -ForegroundColor Cyan
& $py -m uvicorn app.main:app --host 127.0.0.1 --port $Port
