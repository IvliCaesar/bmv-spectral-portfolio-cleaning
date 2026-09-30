$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$paper = Join-Path $root "paper"
Push-Location $paper
try {
    & pdflatex -interaction=nonstopmode -halt-on-error paper_laura_bmv.tex
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex failed on the first pass with exit code $LASTEXITCODE"
    }
    & pdflatex -interaction=nonstopmode -halt-on-error paper_laura_bmv.tex
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex failed on the second pass with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
