$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "setup-gguf.ps1") @args
exit $LASTEXITCODE
