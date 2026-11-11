$env:OMNI_KIT_ACCEPT_EULA='YES'; $env:PYTHONIOENCODING='utf-8'
Set-Location $PSScriptRoot
$py='C:\Users\Owner\isaac_venv\Scripts\python.exe'
foreach($c in @(@('px','zeta66','push'),@('px','zeta41','push','track08'),@('pxnaive','nom','track04'))){
  $a=@('record.py','--engine',$c[0],'--design',$c[1],'--scenario')+@($c[2..($c.Count-1)])
  $p=Start-Process $py -ArgumentList $a -PassThru -WindowStyle Hidden -RedirectStandardOutput ("rec_{0}_{1}.out" -f $c[0],$c[1]) -RedirectStandardError ("rec_{0}_{1}.err" -f $c[0],$c[1])
  $p.PriorityClass='BelowNormal'; $p.WaitForExit()
}
