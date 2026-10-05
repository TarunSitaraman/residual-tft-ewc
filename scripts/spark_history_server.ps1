# Spark History Server for the D22 Spark runs: browse finished runs at http://localhost:18080.
# Needs Hadoop winutils.exe on Windows (HADOOP_HOME) and runs made with --event-log. Reads the event logs that scripts/spark_ageing_features.py writes to data/spark-events. Stop with Ctrl+C.
$repo = Split-Path -Parent $PSScriptRoot
$env:SPARK_HOME = Join-Path $repo ".venv\Lib\site-packages\pyspark"
$events = (Join-Path $repo "data\spark-events") -replace '\\', '/'
New-Item -ItemType Directory -Force (Join-Path $repo "data\spark-events") | Out-Null
$env:SPARK_HISTORY_OPTS = "-Dspark.history.fs.logDirectory=file:///$events -Dspark.history.ui.port=18080"
Write-Host "Spark History Server: http://localhost:18080 (event logs: $events)"
& (Join-Path $env:SPARK_HOME "bin\spark-class.cmd") org.apache.spark.deploy.history.HistoryServer
