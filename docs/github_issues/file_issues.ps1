# Script to copy GitHub issue bodies to clipboard one-at-a-time
# Run this in PowerShell from the repo root (C:\Users\manoh\Desktop\Major Project)

$files = Get-ChildItem "docs/github_issues/*.md" |
    Sort-Object { $_.Name -replace 'issue_(\d+)', '$1' } |
    Where-Object { $_.Name -ne "README.md" }

$i = 1
foreach ($file in $files) {
    # Extract the title from the first line (TITLE: ...)
    $firstLine = Get-Content $file.Path -TotalCount 1
    $title = $firstLine -replace '^TITLE: ', ''

    Write-Host "Issue $i: $title — press Enter to copy to clipboard, Ctrl+C to skip."

    try {
        $key = Read-Host()
    } catch {
        # Ctrl+C — skip this issue
        Write-Host "Skipping Issue $i."
        $i++
        continue
    }
    if ($key -eq "") {

        # Copy the full file content to the clipboard
        Set-Clipboard -Value (Get-Content $file.Path -Raw)

        Write-Host "CLIPBOARD SET — now paste into GitHub issue title/body and submit. Press Enter to continue."
        $key2 = Read-Host()

        $i++
    }
}