@echo off
chcp 65001 >nul
title 耕图 FieldLabel 遥感影像标注工具
cd /d "%~dp0"

echo ================================================
echo   耕图 FieldLabel  遥感影像标注工具
echo ================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 没有检测到 python，请先安装 Python 3.10+ 并勾选 Add to PATH。
  echo        或改用 dist\FieldLabel\启动工具.bat （免安装便携版）
  echo.
  pause
  exit /b 1
)

python -c "import rasterio, PIL, numpy" >nul 2>nul
if errorlevel 1 (
  echo [提示] 依赖不完整，正在安装 rasterio / Pillow / numpy ...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [错误] 依赖安装失败，请手动执行: python -m pip install -r requirements.txt
    pause
    exit /b 1
  )
)

echo 正在启动服务，浏览器将自动打开 http://127.0.0.1:8765/
echo 关闭本窗口即退出程序。
echo.
python annotate.py --port 8765
pause
