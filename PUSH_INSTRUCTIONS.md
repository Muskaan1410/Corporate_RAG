# Push to GitHub - Instructions

## Repository: https://github.com/Muskaan1410/Corporate_RAG

## Step 1: Get Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Name: `Corporate_RAG_Push`
4. Expiration: Choose 90 days (or your preference)
5. **Check the `repo` scope** (this gives full repository access)
6. Click **"Generate token"**
7. **COPY THE TOKEN** - you won't see it again!

## Step 2: Push Your Code

Run this command in PowerShell:

```powershell
git push -u origin main
```

When prompted:
- **Username**: `Muskaan1410`
- **Password**: Paste your Personal Access Token (NOT your GitHub password)

## Alternative: Use Token in URL (One-time)

You can also push with the token in the URL (replace YOUR_TOKEN):

```powershell
git push https://YOUR_TOKEN@github.com/Muskaan1410/Corporate_RAG.git main
```

## After Pushing

Your code will be available at:
**https://github.com/Muskaan1410/Corporate_RAG**

---

## What Will Be Uploaded

✅ All Python code files
✅ All module folders (api, chunking, embedding, etc.)
✅ README.md
✅ requirements.txt
✅ Documentation files
✅ .gitignore

❌ Will NOT upload (thanks to .gitignore):
- `__pycache__/` folders
- `vector_store.*` files
- Virtual environments

