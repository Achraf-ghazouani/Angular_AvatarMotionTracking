@echo off
REM Script d'installation pour Windows
REM Usage: install.bat

echo.
echo 🎭 Avatar IA - Motion Tracking
echo ================================
echo.

REM Vérifier Node.js
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Node.js n'est pas installé
    echo 📥 Téléchargez Node.js depuis: https://nodejs.org/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('node -v') do set NODE_VERSION=%%i
echo ✅ Node.js détecté: %NODE_VERSION%

REM Vérifier npm
where npm >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ npm n'est pas installé
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('npm -v') do set NPM_VERSION=%%i
echo ✅ npm détecté: %NPM_VERSION%

echo.
echo 📦 Installation des dépendances...
call npm install

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors de l'installation
    echo 💡 Essayez: npm install --legacy-peer-deps
    pause
    exit /b 1
)

echo.
echo ✅ Installation réussie!
echo.
echo 🚀 Pour démarrer l'application:
echo    npm start
echo.
echo 📖 Documentation complète: README.md
echo 🆘 En cas de problème: TROUBLESHOOTING.md
echo.
echo Bon développement! 🎉
echo.
pause
