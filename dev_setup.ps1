<#
One-click Jellychat bootstrap (Windows)
• Deletes any existing .venv
• Finds highest Python 3.x:
    – py launcher if present
    – else C:\Python###\python.exe fallbacks
• Creates .venv and installs in editable mode
• No global env persistence — fully repo-isolated
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $ver = (& py -0p) |
               Where-Object { $_ -match '-3\.(\d+)-64' } |
               Sort-Object { [int]$Matches[1] } -Descending |
               Select-Object -First 1
        if ($ver) {
            return "py $($ver -replace ' .*','' -replace '-64$','')"
        }
    }
    $candidates = @(
        'C:\Python312\python.exe',
        'C:\Python311\python.exe',
        'C:\Python310\python.exe'
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    throw "No suitable Python 3.x interpreter found."
}

$python = Find-Python
$env:PIP_NO_NETWORK_SSL_VERIFY = '1'

Write-Host "🐍 Using interpreter: $python"

# ❌ Remove existing virtual environment
if (Test-Path ".venv") {
    Write-Host "🧼 Removing existing .venv..."
    Remove-Item -Recurse -Force ".venv"
}

# ✅ Create new virtual environment
& $python -m venv .venv

# 🔁 Activate and install in editable mode
& .\.venv\Scripts\Activate.ps1
pip install -e .

Write-Host "`n✅ Jellychat .venv ready and activated."
