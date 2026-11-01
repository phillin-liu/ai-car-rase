#Requires -Version 5.1
<#
.SYNOPSIS
    Build the Windows (Nuitka) build of AI Car Race.

.DESCRIPTION
    Compiles main.py with Nuitka on Windows, packages the resulting standalone
    tree (ai-car-rase.exe + every Qt5/SDL2 DLL it needs) into
    dist/ai-car-rase-<version>-windows-x86_64.zip, and optionally emits a
    single self-extracting .exe as well.

    The Linux side of the same job lives in packaging/{nuitka,arch,deb,rpm}/:
    those ship one frozen tree in three package formats, while Windows has to
    be compiled on Windows and ships a portable archive.

.PARAMETER Version
    Version stamped into the executable and the archive name.  Defaults to the
    pkgver in packaging/arch/PKGBUILD (same source of truth as the other
    formats).

.PARAMETER Onefile
    Also produce dist/ai-car-rase-<version>-windows-x86_64.exe, a single
    self-extracting executable (~1 minute extra, reuses Nuitka's cache).
#>
[CmdletBinding()]
param(
    [string] $Version,
    [switch] $Onefile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$outDir  = Join-Path $root 'build\nuitka'
$distDir = Join-Path $root 'dist'
$mainPy  = Join-Path $root 'main.py'

function Get-ProjectVersion {
    if ($Version) { return $Version.TrimStart('v') }
    foreach ($line in Get-Content (Join-Path $root 'packaging\arch\PKGBUILD')) {
        if ($line -match '^pkgver=(.+)$') { return $Matches[1].Trim() }
    }
    throw '无法确定版本号（请使用 -Version 指定）'
}

function Invoke-Nuitka {
    param([string[]] $Arguments)
    Write-Host ">> python -m nuitka $($Arguments -join ' ')"
    & python -m nuitka @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Nuitka 失败（退出码 $LASTEXITCODE）" }
}

$version = Get-ProjectVersion
$numeric = if ($version -match '^\d+(\.\d+){0,3}$') { $version } else { $null }

New-Item -ItemType Directory -Force -Path $outDir, $distDir | Out-Null

# --- Python + Nuitka ------------------------------------------------------
& python -c 'import nuitka' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host '>> 安装 Nuitka ...'
    & python -m pip install --upgrade nuitka ordered-set zstandard
    if ($LASTEXITCODE -ne 0) { throw 'pip install nuitka 失败' }
}

# --- anti-cheat validator -------------------------------------------------
# build.sh embeds the Rust binary as data; without it the game falls back to
# the pure-Python validator, so a failure here only degrades the build.
$acExe = Join-Path $root 'anticheat\target\release\ac-validate.exe'
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host '>> 编译 Rust 反作弊校验器 ...'
    & cargo build --release --locked --manifest-path (Join-Path $root 'anticheat\Cargo.toml')
    if ($LASTEXITCODE -ne 0) { Write-Warning 'cargo build 失败，将使用纯 Python 反作弊校验' }
} else {
    Write-Warning '未找到 cargo，将使用纯 Python 反作弊校验'
}

$dataArgs = @()
if (Test-Path $acExe) {
    $dataArgs += "--include-data-file=$acExe=anticheat/target/release/ac-validate.exe"
}

# Optional weight import paths: skipping them saves ~200 MB and the code has
# an explicit pure-Python fallback.
$nofollow = @(
    '--nofollow-import-to=scipy',
    '--nofollow-import-to=ultralytics',
    '--nofollow-import-to=torch',
    '--nofollow-import-to=torchvision',
    '--nofollow-import-to=matplotlib',
    '--nofollow-import-to=tkinter'
)

$common = @(
    '--assume-yes-for-downloads',
    '--enable-plugin=pyqt5',
    '--include-package=car_game',
    '--include-package-data=car_game',
    '--output-dir=' + $outDir,
    '--output-filename=ai-car-rase.exe',
    '--company-name=AI Car Race',
    '--product-name=AI Car Race',
    '--file-description=AI Car Race',
    '--product-version=' + $(if ($numeric) { $numeric } else { '0.1.0' }),
    '--windows-console-mode=attach',
    '--report=' + (Join-Path $outDir 'report.xml')
) + $nofollow + $dataArgs
if ($numeric) { $common += '--file-version=' + $numeric }

Write-Host ">> Nuitka $(& python -m nuitka --version | Select-Object -First 1) / version=$version"

# --- standalone tree ------------------------------------------------------
Invoke-Nuitka (@('--standalone') + $common + @($mainPy))

$tree = Get-ChildItem -Path $outDir -Filter '*.dist' -Directory |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $tree) { throw 'Nuitka 没有生成 *.dist 目录' }

# --- portable archive -----------------------------------------------------
$zip = Join-Path $distDir "ai-car-rase-$version-windows-x86_64.zip"
Remove-Item -LiteralPath $zip -ErrorAction SilentlyContinue
$sevenZip = 'C:\Program Files\7-Zip\7z.exe'
if (Test-Path $sevenZip) {
    Push-Location $tree.FullName
    try {
        & $sevenZip a -tzip -mx=6 -bso0 -bsp0 $zip '*'
        if ($LASTEXITCODE -ne 0) { throw "7z 失败（退出码 $LASTEXITCODE）" }
    } finally { Pop-Location }
} else {
    Compress-Archive -Path (Join-Path $tree.FullName '*') -DestinationPath $zip `
                     -CompressionLevel Optimal
}

# --- optional single-file executable -------------------------------------
if ($Onefile) {
    Invoke-Nuitka (@('--onefile') + $common + @($mainPy))
    $exe = Join-Path $outDir 'ai-car-rase.exe'
    if (-not (Test-Path $exe)) { throw 'Nuitka --onefile 没有生成 ai-car-rase.exe' }
    Copy-Item $exe (Join-Path $distDir "ai-car-rase-$version-windows-x86_64.exe") -Force
}

Write-Host ''
Write-Host '>> 完成:'
Get-ChildItem $distDir | Format-Table Name, @{N = 'Size'; E = { '{0:N1} MB' -f ($_.Length / 1MB) } }
