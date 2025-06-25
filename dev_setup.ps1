<#
One-click Jellychat bootstrap (Windows)
• Deletes any existing .venv
• Requires Python 3.13+ via py launcher
• Creates .venv and installs in editable mode
• No global env persistence — fully repo-isolated
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $ver = & py -3.13 --version
            if ($LASTEXITCODE -eq 0 -and $ver -match "3\.13") {
                return "py -3.13"
            }
        } catch {
            # Continue to throw below
        }
    }

    throw "❌ Python 3.13 not found via py launcher. Please ensure it's installed and available via 'py -3.13'."
}

$python = Find-Python
$env:PYTHON_EXECUTABLE = $python
$env:PIP_NO_NETWORK_SSL_VERIFY = '1'

Write-Host "🐍 Using interpreter: $python"

# ❌ Remove existing virtual environment
if (Test-Path ".venv") {
    Write-Host "🧼 Removing existing .venv..."
    Remove-Item -Recurse -Force ".venv"
}

# ✅ Create new virtual environment
$parts = $python.Split(" ")
& $parts[0] $parts[1..($parts.Length - 1)] -m venv .venv

# 🔁 Activate and install in editable mode
& .\.venv\Scripts\Activate.ps1
pip install -e .[dev]

Write-Host "`n✅ Jellychat .venv ready and activated."
