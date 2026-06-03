param(
    [string]$InnoCompiler
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$ExePath = Join-Path $ProjectRoot "dist\BreastRiskDesktop\BreastRiskDesktop.exe"
if (-not (Test-Path $ExePath)) {
    & (Join-Path $ProjectRoot "scripts\build_desktop.ps1")
}

$isccPath = $null
if ($InnoCompiler) {
    if (-not (Test-Path $InnoCompiler)) {
        throw "The specified Inno compiler was not found: $InnoCompiler"
    }
    $isccPath = (Resolve-Path $InnoCompiler).Path
}

$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    $commonPaths = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($path in $commonPaths) {
        if ($path -and (Test-Path $path)) {
            $isccPath = (Resolve-Path $path).Path
            break
        }
    }
}
elseif (-not $isccPath) {
    $isccPath = $iscc.Source
}

if (-not $isccPath) {
    throw "Inno Setup compiler ISCC.exe was not found. Find ISCC.exe, then rerun with: .\scripts\build_installer.ps1 -InnoCompiler `"C:\Path\To\ISCC.exe`""
}

$issPath = Join-Path $ProjectRoot "installer\BreastRiskDesktop.iss"
& $isccPath $issPath

$setupPath = Join-Path $ProjectRoot "dist\installer\BreastRiskDesktop_Setup.exe"
if (-not (Test-Path $setupPath)) {
    throw "Installer build failed: $setupPath was not generated."
}

Write-Host "Installer completed: $setupPath"
