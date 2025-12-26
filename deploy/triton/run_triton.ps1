param(
  [string]$TritonTag = "25.10-py3",
  [string]$ModelName = "fashion_mnist",
  [int]$MaxBatchSize = 64
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

python (Join-Path $scriptDir "make_model_repo.py") build `
  --model_name $ModelName `
  --repo_root (Join-Path $repoRoot "deploy\triton\model_repository") `
  --max_batch_size $MaxBatchSize

$modelRepoHost = Join-Path $repoRoot "deploy\triton\model_repository"

Write-Host "Running Triton image: nvcr.io/nvidia/tritonserver:$TritonTag"
Write-Host "Model repo: $modelRepoHost"

docker run --rm --gpus all --shm-size=1g `
  -p 8000:8000 -p 8001:8001 -p 8002:8002 `
  -v "${modelRepoHost}:/models" `
  nvcr.io/nvidia/tritonserver:$TritonTag `
  tritonserver --model-repository=/models
