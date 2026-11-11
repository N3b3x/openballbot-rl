$env:OMNI_KIT_ACCEPT_EULA='YES'; $env:PYTHONIOENCODING='utf-8'
Set-Location $PSScriptRoot
$p=Start-Process 'C:\Users\Owner\isaac_venv\Scripts\python.exe' -ArgumentList 'bench_px.py','--n','4096','--device','gpu','--ctrl-steps','10','--out','..\..\outputs\study\isaac_port\bench_px_gpu_4096.json' -PassThru -WindowStyle Hidden -RedirectStandardOutput bench4096.out -RedirectStandardError bench4096.err
$p.PriorityClass='BelowNormal'; $p.WaitForExit()
