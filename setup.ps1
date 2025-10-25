# ===========================================
# Sentria API Local Setup Script (Windows)
# ===========================================

Write-Host "`n🚀 Starting Sentria API setup..." -ForegroundColor Cyan

# --- Step 1: Check Python 3.10 ---
if (-not (py -3.10 --version 2>$null)) {
    Write-Host "❌ Python 3.10 not found. Please install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# --- Step 2: Create & activate venv ---
Write-Host "🐍 Creating virtual environment..." -ForegroundColor Yellow
py -3.10 -m venv .venv

Write-Host "⚙️  Activating venv..." -ForegroundColor Yellow
try {
    .\.venv\Scripts\Activate.ps1
} catch {
    Write-Host "⚠️  Activation blocked. Temporarily bypassing execution policy..." -ForegroundColor DarkYellow
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    .\.venv\Scripts\Activate.ps1
}

# --- Step 3: Install dependencies ---
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt

# --- Step 4: .env file check ---
$envPath = ".\.env"
if (-not (Test-Path $envPath)) {
    Write-Host "📝 Creating sample .env file..." -ForegroundColor Yellow
    @"
LIVEKIT_URL=wss://your-livekit-domain.livekit.cloud
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret

BASETEN_API_KEY=your_baseten_key
BASETEN_THEFT_ENDPOINT=https://your-endpoint/predict
BASETEN_WEAPON_ENDPOINT=https://your-endpoint/predict
BASETEN_FACE_ENDPOINT=https://your-endpoint/predict
"@ | Out-File -Encoding UTF8 .env
    Write-Host "✅ .env created — remember to update it with real keys!" -ForegroundColor Green
} else {
    Write-Host "✅ .env file found." -ForegroundColor Green
}

# --- Step 5: Run tests ---
Write-Host "🧪 Running tests..." -ForegroundColor Yellow
pytest -q

# --- Step 6: Launch API ---
Write-Host "`n🌐 Starting API on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
