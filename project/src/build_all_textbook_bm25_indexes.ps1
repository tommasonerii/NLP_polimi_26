$ErrorActionPreference = "Stop"

$books = @(
    @{
        book_id = "introductory_statistics_2e"
        book_title = "Introductory Statistics 2e"
        source = "openstax_introductory_statistics_2e"
        pdf = "data\maths\introductory-statistics-2e_-_WEB.pdf"
        start_page = 17
    },
    @{
        book_id = "algebra_trigonometry_2e"
        book_title = "Algebra and Trigonometry 2e"
        source = "openstax_algebra_trigonometry_2e"
        pdf = "data\maths\algebra-and-trigonometry-2e_-_WEB.pdf"
        start_page = 18
    },
    @{
        book_id = "calculus_volume_1"
        book_title = "Calculus Volume 1"
        source = "openstax_calculus_volume_1"
        pdf = "data\maths\calculus-volume-1_-_WEB.pdf"
        start_page = 15
    },
    @{
        book_id = "discrete_math_open_intro"
        book_title = "Discrete Mathematics: An Open Introduction"
        source = "discrete_math_open_intro"
        pdf = "data\maths\dmoi4.pdf"
        start_page = 25
    },
    @{
        book_id = "abstract_algebra_judson"
        book_title = "Abstract Algebra: Theory and Applications"
        source = "abstract_algebra_judson"
        pdf = "data\maths\AbstractAlgebra.pdf"
        start_page = 14
    },
    @{
        book_id = "basic_analysis_1"
        book_title = "Basic Analysis I"
        source = "basic_analysis_1"
        pdf = "data\maths\realanalysis.pdf"
        start_page = 23
    },
    @{
        book_id = "topology_without_tears"
        book_title = "Topology Without Tears"
        source = "topology_without_tears"
        pdf = "data\maths\topology.pdf"
        start_page = 13
    }
)

Write-Host "============================================================"
Write-Host "[DEBUG] Starting textbook BM25 build"
Write-Host "[DEBUG] Current directory: $(Get-Location)"
Write-Host "[DEBUG] Python executable:"
python -c "import sys; print(sys.executable)"
Write-Host "[DEBUG] Python version:"
python --version
Write-Host "[DEBUG] Number of books: $($books.Count)"
Write-Host "============================================================"

New-Item -ItemType Directory -Force -Path "data\chunks" | Out-Null
New-Item -ItemType Directory -Force -Path "data\indexes" | Out-Null

foreach ($book in $books) {
    $bookId = $book.book_id
    $chunkPath = "data\chunks\$($bookId)_200w.jsonl"
    $indexPath = "data\indexes\$($bookId)_200w_section2_stop_ngram2_bm25.joblib"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[DEBUG] Book ID:      $bookId"
    Write-Host "[DEBUG] Title:        $($book.book_title)"
    Write-Host "[DEBUG] Source:       $($book.source)"
    Write-Host "[DEBUG] PDF:          $($book.pdf)"
    Write-Host "[DEBUG] Start page:   $($book.start_page)"
    Write-Host "[DEBUG] Chunk output: $chunkPath"
    Write-Host "[DEBUG] Index output: $indexPath"
    Write-Host "============================================================"

    if (-Not (Test-Path $book.pdf)) {
        throw "[ERROR] PDF not found: $($book.pdf)"
    }

    if (-Not (Test-Path "project\src\prepare_textbook_chunks.py")) {
        throw "[ERROR] Missing script: project\src\prepare_textbook_chunks.py"
    }

    if (-Not (Test-Path "project\src\build_retrieval_index.py")) {
        throw "[ERROR] Missing script: project\src\build_retrieval_index.py"
    }

    Write-Host "[DEBUG] Step 1/2: preparing chunks..."
    $chunkStart = Get-Date

    python project\src\prepare_textbook_chunks.py `
        --pdf $book.pdf `
        --book-id $book.book_id `
        --book-title $book.book_title `
        --source $book.source `
        --output $chunkPath `
        --chunk-words 200 `
        --overlap-words 50 `
        --start-page $book.start_page

    if ($LASTEXITCODE -ne 0) {
        throw "[ERROR] Chunking failed for $bookId with exit code $LASTEXITCODE"
    }

    $chunkEnd = Get-Date
    $chunkSeconds = [math]::Round(($chunkEnd - $chunkStart).TotalSeconds, 2)

    if (-Not (Test-Path $chunkPath)) {
        throw "[ERROR] Chunk file was not created: $chunkPath"
    }

    $chunkCount = (Get-Content $chunkPath).Count
    $chunkSizeMb = [math]::Round((Get-Item $chunkPath).Length / 1MB, 2)

    Write-Host "[DEBUG] Chunking completed for $bookId"
    Write-Host "[DEBUG] Chunk count: $chunkCount"
    Write-Host "[DEBUG] Chunk file size: $chunkSizeMb MB"
    Write-Host "[DEBUG] Chunking time: $chunkSeconds sec"

    Write-Host "[DEBUG] First chunk preview:"
    Get-Content $chunkPath -TotalCount 1

    Write-Host ""
    Write-Host "[DEBUG] Step 2/2: building BM25 index..."
    $indexStart = Get-Date

    python project\src\build_retrieval_index.py $chunkPath `
        -o $indexPath `
        --kind bm25 `
        --title-repeat 2 `
        --bm25-remove-stopwords `
        --bm25-ngram-max 2 `
        --compress 3

    if ($LASTEXITCODE -ne 0) {
        throw "[ERROR] BM25 indexing failed for $bookId with exit code $LASTEXITCODE"
    }

    $indexEnd = Get-Date
    $indexSeconds = [math]::Round(($indexEnd - $indexStart).TotalSeconds, 2)

    if (-Not (Test-Path $indexPath)) {
        throw "[ERROR] Index file was not created: $indexPath"
    }

    $indexSizeMb = [math]::Round((Get-Item $indexPath).Length / 1MB, 2)

    Write-Host "[DEBUG] BM25 completed for $bookId"
    Write-Host "[DEBUG] Index file size: $indexSizeMb MB"
    Write-Host "[DEBUG] Indexing time: $indexSeconds sec"
    Write-Host "[OK] Finished $bookId"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[OK] Done. Built all textbook BM25 indexes."
Write-Host "============================================================"

Write-Host "[DEBUG] Generated textbook index files:"
Get-ChildItem data\indexes\*_200w_section2_stop_ngram2_bm25.joblib |
    Select-Object Name, Length, LastWriteTime