# Spark History Server for the D22 Spark runs: browse finished runs in the official Spark UI at http://localhost:18080.
# Shows runs made with  scripts/spark_ageing_features.py --event-log  (event logs in data/spark-events).
# Windows needs Hadoop's winutils.exe: here C:\Users\Tarun\tools\hadoop-3.3.6\bin\winutils.exe (from
# github.com/cdarlint/winutils, hadoop-3.3.6; SHA-256 496a591eb1e67df2a620f710d529ba6ddfe1c19149e6647cc4e320bb0efd8553).
# The server also needs hadoop.dll (same source; SHA-256 d7ab36a68518748cef142be2da5069b4c763c2cd764c1d2e6ac48c7200405be3) in
# HADOOP_HOME\bin, loaded through java.library.path for this process only. Set HADOOP_HOME to override. Stop with Ctrl+C (or stop the java process).
$repo = Split-Path -Parent $PSScriptRoot
if (-not $env:HADOOP_HOME) { $env:HADOOP_HOME = "C:\Users\Tarun\tools\hadoop-3.3.6" }
foreach ($f in "winutils.exe", "hadoop.dll") { if (-not (Test-Path (Join-Path $env:HADOOP_HOME "bin\$f"))) { throw "$f not found under HADOOP_HOME=$env:HADOOP_HOME\bin" } }
$env:SPARK_HOME = Join-Path $repo ".venv\Lib\site-packages\pyspark"
# spark-class.cmd splits options at spaces and the History Server does not decode %20, so a repository path with spaces
# ("ML handover") is read through a space-free directory junction to data\spark-events, created here if missing.
$events = (Join-Path $repo "data\spark-events")
if ($events -match ' ') {
    $alias = Join-Path $env:USERPROFILE "spark-events"
    if (-not (Test-Path -LiteralPath $alias)) { New-Item -ItemType Junction -Path $alias -Target $events | Out-Null }
    $events = $alias
}
$events = $events -replace '\\', '/'
New-Item -ItemType Directory -Force (Join-Path $repo "data\spark-events") | Out-Null
$env:SPARK_HISTORY_OPTS = "-Dspark.history.fs.logDirectory=file:///$events -Dspark.history.ui.port=18080 -Dspark.history.fs.update.interval=10s -Djava.library.path=$($env:HADOOP_HOME -replace '\\', '/')/bin"
Write-Host "Spark History Server: http://localhost:18080 (event logs: $events)"
& (Join-Path $env:SPARK_HOME "bin\spark-class.cmd") org.apache.spark.deploy.history.HistoryServer


