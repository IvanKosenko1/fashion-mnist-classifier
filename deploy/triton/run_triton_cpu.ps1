param(
  [string]$TritonTag = "25.10-py3",
  [string]$ModelName = "fashion_mnist",
  [int]$MaxBatchSize = 64
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# build model repo from ONNX
python (Join-Path $scriptDir "make_model_repo.py") build `
  --model_name $ModelName `
  --repo_root (Join-Path $repoRoot "deploy\triton\model_repository") `
  --max_batch_size $MaxBatchSize

# copy external onnx data if exists (model.onnx.data)
$onnxDir  = Join-Path $repoRoot "artifacts\onnx"
$onnxFile = Get-ChildItem -Path $onnxDir -Filter *.onnx -File | Select-Object -First 1
if (-not $onnxFile) { throw "No .onnx found in $onnxDir" }

$dataFile = Join-Path $onnxDir ($onnxFile.Name + ".data")
$versionDir = Join-Path $repoRoot ("deploy\triton\model_repository\{0}\1" -f $ModelName)

if (Test-Path $dataFile) {
  Copy-Item $dataFile (Join-Path $versionDir "model.onnx.data") -Force
}

# force CPU instances (no GPU required)
$configPath = Join-Path $repoRoot ("deploy\triton\model_repository\{0}\config.pbtxt" -f $ModelName)
(Get-Content $configPath -Raw) -replace "KIND_GPU", "KIND_CPU" | Set-Content -Encoding UTF8 $configPath

$modelRepoHost = Join-Path $repoRoot "deploy\triton\model_repository"

docker run --rm --shm-size=1g `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${modelRepoHost}:/models" `
  nvcr.io/nvidia/tritonserver:$TritonTag `
  tritonserver --model-repository=/models
