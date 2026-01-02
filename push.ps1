# Push to GitHub using Personal Access Token
# Replace YOUR_TOKEN_HERE with your actual token

$token = "YOUR_TOKEN_HERE"  # Paste your token here
$repo = "https://github.com/Muskaan1410/Corporate_RAG.git"

# Add files if needed
git add .

# Commit if there are changes
$status = git status --porcelain
if ($status) {
    git commit -m "Add push instructions"
}

# Push using token
$remote = "https://$token@github.com/Muskaan1410/Corporate_RAG.git"
git push $remote main

Write-Host "Done! Check: https://github.com/Muskaan1410/Corporate_RAG" -ForegroundColor Green


