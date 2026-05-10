$ErrorActionPreference = "Stop"

$PythonExe = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$books = @(
    @{
        book_id = "introductory_statistics_2e"
        source = "textbook_introductory_statistics"
    },
    @{
        book_id = "algebra_trigonometry_2e"
        source = "textbook_algebra_trigonometry"
    },
    @{
        book_id = "calculus_volume_1"
        source = "textbook_calculus_volume_1"
    },
    @{
        book_id = "discrete_math_open_intro"
        source = "textbook_discrete_math"
    },
    @{
        book_id = "abstract_algebra_judson"
        source = "textbook_abstract_algebra"
    },
    @{
        book_id = "basic_analysis_1"
        source = "textbook_basic_analysis"
    },
    @{
        book_id = "topology_without_tears"
        source = "textbook_topology"
    }
)

Write-Host "============================================================"
Write-Host "[DEBUG] Starting textbook dense HNSW build"
Write-Host "[DEBUG] Current directory: $(Get-Location)"
Write-Host "[DEBUG] Python executable command: $PythonExe"
Write-Host "[DEBUG] Number of books: $($books.Count)"
Write-Host "============================================================"

New-Item -ItemType Directory -Force -Path "data\indexes" | Out-Null

foreach ($book in $books) {
    $bookId = $book.book_id
    $source = $book.source
    $chunkPath = "data\chunks\$($bookId)_200w.jsonl"
    $indexPath = "data\indexes\$($bookId)_200w_dense_hnsw.index"
    $metaPath = "data\indexes\$($bookId)_200w_dense_meta.joblib"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[DEBUG] Book ID:      $bookId"
    Write-Host "[DEBUG] Source:       $source"
    Write-Host "[DEBUG] Chunk input:  $chunkPath"
    Write-Host "[DEBUG] Index output: $indexPath"
    Write-Host "[DEBUG] Meta output:  $metaPath"
    Write-Host "============================================================"

    if (-Not (Test-Path $chunkPath)) {
        throw "[ERROR] Chunk file not found: $chunkPath. Run project\src\build_all_textbook_bm25_indexes.ps1 first."
    }

    if (-Not (Test-Path "project\src\build_dense_hnsw_index.py")) {
        throw "[ERROR] Missing script: project\src\build_dense_hnsw_index.py"
    }

    $chunkCount = (Get-Content $chunkPath).Count
    $chunkSizeMb = [math]::Round((Get-Item $chunkPath).Length / 1MB, 2)
    Write-Host "[DEBUG] Chunk count: $chunkCount"
    Write-Host "[DEBUG] Chunk file size: $chunkSizeMb MB"

    $indexStart = Get-Date

    & $PythonExe project\src\build_dense_hnsw_index.py $chunkPath `
        --index-output $indexPath `
        --meta-output $metaPath `
        --source $source `
        --model-name "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" `
        --device cpu `
        --add-batch-size 2048 `
        --encode-batch-size 32 `
        --title-repeat 1 `
        --m 32 `
        --ef-construction 200 `
        --ef-search 128 `
        --compress 3

    if ($LASTEXITCODE -ne 0) {
        throw "[ERROR] Dense indexing failed for $bookId with exit code $LASTEXITCODE"
    }

    $indexEnd = Get-Date
    $indexSeconds = [math]::Round(($indexEnd - $indexStart).TotalSeconds, 2)

    if (-Not (Test-Path $indexPath)) {
        throw "[ERROR] Dense index file was not created: $indexPath"
    }
    if (-Not (Test-Path $metaPath)) {
        throw "[ERROR] Dense meta file was not created: $metaPath"
    }

    $indexSizeMb = [math]::Round((Get-Item $indexPath).Length / 1MB, 2)
    $metaSizeMb = [math]::Round((Get-Item $metaPath).Length / 1MB, 2)

    Write-Host "[DEBUG] Dense completed for $bookId"
    Write-Host "[DEBUG] Index file size: $indexSizeMb MB"
    Write-Host "[DEBUG] Meta file size: $metaSizeMb MB"
    Write-Host "[DEBUG] Indexing time: $indexSeconds sec"
    Write-Host "[OK] Finished $bookId"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[OK] Done. Built all textbook dense HNSW indexes."
Write-Host "============================================================"

Write-Host "[DEBUG] Generated textbook dense index files:"
Get-ChildItem data\indexes\*_200w_dense_hnsw.index, data\indexes\*_200w_dense_meta.joblib |
    Select-Object Name, Length, LastWriteTime
