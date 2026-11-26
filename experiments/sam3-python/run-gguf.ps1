param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Image,

    [string]$Model,

    [string]$OutputDirectory,

    [ValidateSet("auto", "cuda", "vulkan", "cpu")]
    [string]$Device = "auto",

    [string[]]$Prompt
)

$ErrorActionPreference = "Stop"
$experimentRoot = $PSScriptRoot
$modelDirectory = Join-Path $experimentRoot "models"
$runtimeBuild = Join-Path $experimentRoot ".deps\sam3-ggml\build-inlay"
$executable = Get-ChildItem -Path (Join-Path $runtimeBuild "examples") -Filter "sam3_vis.exe" -File -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1

if (-not $executable) {
    throw "sam3_vis.exe was not found. Run .\setup.ps1 first to build the native GGUF runtime."
}

$imagePath = (Resolve-Path -LiteralPath $Image).Path
if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf)) {
    throw "Input image was not found: $Image"
}

if (-not $Model) {
    $preferredModels = @(
        "sam3.1-f16.gguf",
        "sam3.1-q8_0.gguf",
        "sam3-f16.gguf",
        "sam3-q8_0.gguf"
    )
    foreach ($candidate in $preferredModels) {
        $candidatePath = Join-Path $modelDirectory $candidate
        if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
            $Model = $candidatePath
            break
        }
    }
}

if (-not $Model) {
    $availableModels = @(Get-ChildItem -Path $modelDirectory -Filter "*.gguf" -File -ErrorAction SilentlyContinue)
    if ($availableModels.Count -eq 1) {
        $Model = $availableModels[0].FullName
    }
    elseif ($availableModels.Count -gt 1) {
        $names = ($availableModels | ForEach-Object { $_.Name }) -join ", "
        throw "More than one GGUF model was found ($names). Specify one with -Model."
    }
    else {
        throw "No GGUF model found in $modelDirectory. Put your SAM 3.1 .gguf file there or pass -Model <path>."
    }
}

$modelPath = (Resolve-Path -LiteralPath $Model).Path
if ([IO.Path]::GetExtension($modelPath) -ne ".gguf") {
    throw "Expected a .gguf model file, received: $modelPath"
}

$imageInfo = Get-Item -LiteralPath $imagePath
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $imageInfo.DirectoryName "$($imageInfo.BaseName).sam3-gguf"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$outputRoot = (Resolve-Path -LiteralPath $OutputDirectory).Path

$prompts = if ($Prompt -and $Prompt.Count -gt 0) {
    @($Prompt)
}
else {
    @(
        "the complete foreground photograph or image panel, including its rectangular edges",
        "the entire rectangular picture displayed in front of the background",
        "the full image card or photo viewer panel, not the person or object inside it"
    )
}

$results = @()
$failed = $false
for ($index = 0; $index -lt $prompts.Count; $index++) {
    $currentPrompt = $prompts[$index]
    $slug = ($currentPrompt.ToLowerInvariant() -replace "[^a-z0-9]+", "-").Trim("-")
    if ($slug.Length -gt 48) { $slug = $slug.Substring(0, 48).TrimEnd("-") }

    $prefix = "{0:D2}-{1}" -f ($index + 1), $slug
    $outputPath = Join-Path $outputRoot "$prefix.png"
    $logPath = Join-Path $outputRoot "$prefix.log"

    Write-Host ""
    Write-Host "[$($index + 1)/$($prompts.Count)] $currentPrompt"
    $nativeOutput = & $executable.FullName --model $modelPath --image $imagePath --prompt $currentPrompt --out $outputPath --device $Device 2>&1
    $exitCode = $LASTEXITCODE
    $nativeOutput | Set-Content -LiteralPath $logPath -Encoding utf8
    $nativeOutput | ForEach-Object { Write-Host $_ }

    $success = $exitCode -eq 0 -and (Test-Path -LiteralPath $outputPath -PathType Leaf)
    if (-not $success) { $failed = $true }

    $results += [pscustomobject]@{
        prompt = $currentPrompt
        output = $outputPath
        log = $logPath
        exit_code = $exitCode
        success = $success
    }
}

$report = [pscustomobject]@{
    input = $imagePath
    model = $modelPath
    device = $Device
    results = $results
    note = "These are model-generated segmentation overlays, not verified crop correctness. Inspect whether each mask follows the outer photo panel rather than its contents."
}
$reportPath = Join-Path $outputRoot "report.json"
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host ""
Write-Host "Diagnostics: $outputRoot"
Write-Host "Report: $reportPath"
if ($failed) {
    throw "One or more prompts failed. Check the corresponding .log files."
}
