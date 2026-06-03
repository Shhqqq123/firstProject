param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if ($Clean) {
    Get-Process -Name "BreastRiskDesktop" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2

    if (Test-Path "build") {
        Remove-Item -LiteralPath "build" -Recurse -Force
    }
    if (Test-Path "dist\BreastRiskDesktop") {
        Remove-Item -LiteralPath "dist\BreastRiskDesktop" -Recurse -Force
    }
}

python -m PyInstaller BreastRiskDesktop.spec --noconfirm

$ExePath = Join-Path $ProjectRoot "dist\BreastRiskDesktop\BreastRiskDesktop.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build failed: $ExePath was not generated."
}

Write-Host "Build completed: $ExePath"
