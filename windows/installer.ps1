# Installation complete du projet sous Windows (a lancer une seule fois)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== Verification de Python ==" -ForegroundColor Cyan
python --version
if ($LASTEXITCODE -ne 0) { Write-Host "Python introuvable : installez Python 3.11 depuis python.org" -ForegroundColor Red; exit 1 }

Write-Host "== Verification de ffmpeg ==" -ForegroundColor Cyan
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ffmpeg absent : installation via winget..." -ForegroundColor Yellow
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Write-Host "Fermez puis rouvrez PowerShell apres cette installation, puis relancez ce script." -ForegroundColor Yellow
}

Write-Host "== Verification d'Ollama (sert le modele Gemma) ==" -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama absent : installation via winget..." -ForegroundColor Yellow
    winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    try { Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null }
    catch { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep -Seconds 5 }
    Write-Host "Telechargement de Gemma (gemma4:e4b, plusieurs Go)..." -ForegroundColor Cyan
    ollama pull gemma4:e4b
} else {
    Write-Host "Ollama installe : fermez puis rouvrez PowerShell, puis relancez ce script." -ForegroundColor Yellow
}

Write-Host "== Environnement virtuel ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1

Write-Host "== Dependances (plusieurs minutes) ==" -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-dev.txt

Write-Host "== Tests en mode simulation ==" -ForegroundColor Cyan
$env:WVC_MOCK_MODELS = "1"
python -m pytest -q
Remove-Item Env:WVC_MOCK_MODELS

Write-Host ""
Write-Host "Installation terminee. Lancez maintenant : windows\lancer.bat" -ForegroundColor Green
