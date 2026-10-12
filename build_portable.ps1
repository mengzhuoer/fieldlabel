# =====================================================================
#  一键打包「免安装便携版」——目标电脑无需安装 Python / GDAL
#  用法： 右键「使用 PowerShell 运行」  或  .\build_portable.ps1
# =====================================================================
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $here

Write-Host "== 1/5 检查 Python ==" -ForegroundColor Cyan
# 可用环境变量 PA_PYTHON 指定解释器（PATH 里没有 python 时）：
#   $env:PA_PYTHON = "D:\Claude\Python310\python.exe"; .\build_portable.ps1
if ($env:PA_PYTHON) {
  $pyExe = $env:PA_PYTHON
  if (-not (Test-Path $pyExe)) { throw "PA_PYTHON 指定的解释器不存在: $pyExe" }
} else {
  $py = (Get-Command python -ErrorAction SilentlyContinue)
  if (-not $py) { throw "未找到 python，请安装 Python 3.10+ 并加入 PATH，或用环境变量 PA_PYTHON 指定解释器" }
  $pyExe = $py.Source
}
& $pyExe -c "import sys; assert sys.version_info >= (3,9); print(sys.version)"

Write-Host "== 2/5 检查 / 安装 PyInstaller ==" -ForegroundColor Cyan
$ErrorActionPreference = "Continue"          # 检测命令的 stderr 不应中断脚本
& $pyExe -m PyInstaller --version 2>$null
$hasPI = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if (-not $hasPI) {
  & $pyExe -m pip install --disable-pip-version-check pyinstaller
  if ($LASTEXITCODE -ne 0) { throw "PyInstaller 安装失败" }
}

Write-Host "== 3/5 清理旧产物 ==" -ForegroundColor Cyan
foreach ($p in @("$here\build", "$here\dist", "$here\FieldLabel.spec")) {
  if (Test-Path $p) { Remove-Item $p -Recurse -Force }
}

Write-Host "== 4/5 开始构建（约 1-2 分钟）==" -ForegroundColor Cyan
# 注意：--exclude-module 那串是关键——scipy 的 array_api_compat 会静态引用 torch/transformers/cv2
# 等大依赖（运行时是受保护的可选导入），不排除会把 4GB 的 ML 生态打进包里。
& $pyExe -m PyInstaller --noconfirm --clean --onedir --name FieldLabel `
  --add-data "$here\index.html;." `
  --collect-all rasterio `
  --exclude-module tkinter --exclude-module matplotlib `
  --exclude-module PyQt5 --exclude-module PySide2 --exclude-module pytest `
  --exclude-module jedi --exclude-module IPython --exclude-module chardet `
  --exclude-module dask --exclude-module pandas `
  --exclude-module torch --exclude-module torchvision --exclude-module functorch `
  --exclude-module transformers --exclude-module huggingface_hub `
  --exclude-module safetensors --exclude-module einops --exclude-module timm `
  --exclude-module cv2 --exclude-module sklearn --exclude-module onnxruntime `
  --exclude-module tensorflow --exclude-module keras --exclude-module jax `
  --exclude-module jaxlib `
  "$here\annotate.py"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败" }

Write-Host "== 5/5 生成启动器并打包 zip ==" -ForegroundColor Cyan
$appDir = "$here\dist\FieldLabel"
if (-not (Test-Path $appDir)) { throw "未找到构建产物 $appDir" }

# 防火墙自动放行脚本（按解压目录动态建规则；已放行则秒退，未放行弹一次 UAC）
# 带 BOM 的 UTF8 —— PowerShell 5.1 读无 BOM 会把中文按 ANSI 解码导致乱码
$fw = @'
# ============================================================
#  首次运行自动放行局域网访问（按当前解压目录的 exe 路径建规则）
#  已放行 → 静默退出；未放行 → 弹一次 UAC 添加后退出
#  单机使用可直接取消 UAC，不影响本机启动
# ============================================================
$exe = Join-Path $PSScriptRoot "FieldLabel.exe"

# 逃生舱：PA_SKIP_FW=1 跳过整个放行流程（单机/CI/无 UAC 环境用）
if ($env:PA_SKIP_FW -eq "1") { exit 0 }

function Test-LanAllowed {
  $fs = Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue |
        Where-Object { $_.Program -eq $exe }
  foreach ($f in $fs) {
    $rule = $f | Get-NetFirewallRule -ErrorAction SilentlyContinue | Where-Object {
      $_.Direction -eq "Inbound" -and $_.Action -eq "Allow" -and $_.Enabled -eq "True" }
    if ($rule) { return $true }
  }
  return $false
}

if (Test-LanAllowed) { exit 0 }

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$pr = New-Object Security.Principal.WindowsPrincipal($id)
if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  try {
    Start-Process powershell -Verb RunAs `
      -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSCommandPath`"" `
      -Wait -ErrorAction Stop
  } catch {
    Write-Host "  已跳过防火墙授权：单机使用不受影响；局域网访问请见使用说明「故障排查」。"
  }
  exit 0
}

try {
  New-NetFirewallRule -DisplayName "FieldLabel (LAN)" -Direction Inbound `
    -Program $exe -Action Allow -Profile Any -Enabled True -ErrorAction Stop | Out-Null
  Write-Host "  局域网访问已放行。"
} catch {
  Write-Host ("  防火墙放行失败: " + $_.Exception.Message)
}
'@
[IO.File]::WriteAllText("$appDir\放行防火墙.ps1", $fw, (New-Object System.Text.UTF8Encoding($true)))

# 启动器：先自动放行局域网，再启动；%* 透传端口等参数（如 --local-only / --port 8799）
$launcher = @'
@echo off
chcp 65001 >nul
title 耕图 FieldLabel 遥感标注工具 (便携版)
cd /d "%~dp0"
echo ================================================
echo   耕图 FieldLabel  遥感影像标注工具
echo ================================================
echo   [1/2] 检查局域网访问（首次运行会弹授权窗口，请点"是"）...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0放行防火墙.ps1"
echo   [2/2] 启动服务（浏览器将自动打开）...
echo   本机:   http://127.0.0.1:8765/
echo   局域网: http://本机IP:8765/   （IP 见下方启动日志）
echo   关闭本窗口即退出程序。
echo.
FieldLabel.exe --port 8765 %*
pause
'@
# 无 BOM UTF8 + CRLF 行尾 —— cmd 读带 BOM 的 bat 会把 BOM 当首字符报错；
# 纯 LF 行尾在 chcp 65001 + 多字节中文下会让 cmd 解析错乱（行截断），必须 CRLF
$batText = $launcher -replace "`r?`n", "`r`n"
[IO.File]::WriteAllText("$appDir\启动工具.bat", $batText, (New-Object System.Text.UTF8Encoding($false)))

# 详细说明文档随包分发（README → 使用说明.md）
if (Test-Path "$here\README.md") {
  Copy-Item "$here\README.md" "$appDir\使用说明.md" -Force
  Write-Host "已附带 使用说明.md"
}

$zip = "$here\FieldLabel_v1.0_portable.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Write-Host "正在压缩 $zip ..."
Compress-Archive -Path "$appDir\*" -DestinationPath $zip -CompressionLevel Optimal

Write-Host ""
Write-Host "构建完成！" -ForegroundColor Green
Write-Host "  可运行目录 : $appDir"
Write-Host "  便携压缩包 : $zip"
Write-Host "把 zip 拷到别的电脑解压后，双击「启动工具.bat」即可使用。"
