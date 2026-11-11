Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*--name gdiag_s0*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
