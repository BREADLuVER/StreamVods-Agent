# StreamSniped GPU Orchestrator Setup
param(
    [switch]$Status,
    [switch]$Force,
    [switch]$Verbose,
    [switch]$Stop,
    [switch]$LowImpact,
    [int]$GpuThreshold,
    [int]$MemThreshold,
    [int]$CpuThreshold,
    [int]$SleepSeconds
)

# Load environment variables from config file
Write-Host "🔧 Loading environment variables..." -ForegroundColor Cyan
if (Test-Path "setup_env.ps1") {
    . .\setup_env.ps1
} else {
    Write-Host "❌ setup_env.ps1 not found. Please run setup_env.ps1 first." -ForegroundColor Red
    exit 1
}

$daemonScript = "aws-scripts/gpu_orchestrator_daemon.py"
$logFile = "logs/gpu_orchestrator.log"

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}

function Get-OrchestratorProcesses {
    $processes = @()
    try {
        $pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue
        foreach ($proc in $pythonProcs) {
            try {
                $cmdLine = (Get-WmiObject -Class Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
                if ($cmdLine -and $cmdLine -like "*gpu_orchestrator_daemon.py*") {
                    $processes += $proc
                }
            } catch { }
        }
    } catch { }
    return $processes
}

function Stop-Orchestrator {
    $processes = Get-OrchestratorProcesses
    if ($processes.Count -gt 0) {
        Write-Host "🛑 Stopping $($processes.Count) orchestrator process(es)..." -ForegroundColor Yellow
        foreach ($proc in $processes) {
            Stop-Process -Id $proc.Id -Force
            Write-Host "   Stopped PID $($proc.Id)" -ForegroundColor Green
        }
        Start-Sleep 2
        return $true
    }
    return $false
}

function Show-Status {
    $processes = Get-OrchestratorProcesses
    if ($processes.Count -gt 0) {
        Write-Host " Orchestrator is running ($($processes.Count) process(es))" -ForegroundColor Green
        foreach ($proc in $processes) {
            Write-Host "   PID: $($proc.Id)" -ForegroundColor Cyan
        }
        
        # Show recent logs
        if (Test-Path $logFile) {
            Write-Host "`n📝 Recent logs:" -ForegroundColor Yellow
            Get-Content $logFile -Tail 10 | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
        }
        
        # Show queue status
        try {
            $clipQueue = aws sqs get-queue-attributes --queue-url $env:CLIP_QUEUE_URL --attribute-names ApproximateNumberOfMessages --output json 2>$null | ConvertFrom-Json
            $renderQueue = aws sqs get-queue-attributes --queue-url $env:RENDER_QUEUE_URL --attribute-names ApproximateNumberOfMessages --output json 2>$null | ConvertFrom-Json
            
            Write-Host "`n Queue Status:" -ForegroundColor Yellow
            Write-Host "   Clip Queue: $($clipQueue.Attributes.ApproximateNumberOfMessages) pending" -ForegroundColor Cyan
            Write-Host "   Render Queue: $($renderQueue.Attributes.ApproximateNumberOfMessages) pending" -ForegroundColor Cyan
        } catch {
            Write-Host "   Could not get queue status" -ForegroundColor Yellow
        }
    } else {
        Write-Host "X Orchestrator is not running" -ForegroundColor Red
    }
}

# Main execution
if ($Status) {
    Show-Status
    exit 0
}

if ($Stop) {
    if (Stop-Orchestrator) {
        Write-Host " Orchestrator stopped" -ForegroundColor Green
    } else {
        Write-Host "ℹ️ No orchestrator processes found" -ForegroundColor Yellow
    }
    exit 0
}

# Check if script exists
if (-not (Test-Path $daemonScript)) {
    Write-Host "X Daemon script not found: $daemonScript" -ForegroundColor Red
    exit 1
}

# Stop existing processes if requested
if ($Force) {
    Stop-Orchestrator | Out-Null
}

# Check for existing processes
$existingProcesses = Get-OrchestratorProcesses
if ($existingProcesses.Count -gt 0) {
    Write-Host " Orchestrator is already running" -ForegroundColor Yellow
    Show-Status
    exit 0
}

# Start the orchestrator
Write-Host "🚀 Starting GPU Orchestrator..." -ForegroundColor Cyan

$processArgs = @($daemonScript)

# Add required queue URLs if available
if ($env:CLIP_QUEUE_URL) {
    $processArgs += @("--clip-queue-url", $env:CLIP_QUEUE_URL)
} else {
    Write-Host "⚠️  CLIP_QUEUE_URL not set - orchestrator may not process clip jobs" -ForegroundColor Yellow
}

if ($env:RENDER_QUEUE_URL) {
    $processArgs += @("--render-queue-url", $env:RENDER_QUEUE_URL)
} else {
    Write-Host "⚠️  RENDER_QUEUE_URL not set - orchestrator may not process render jobs" -ForegroundColor Yellow
}

# Optional FULL queue (full local workflow)
if ($env:FULL_QUEUE_URL) {
    $processArgs += @("--full-queue-url", $env:FULL_QUEUE_URL)
}

# Apply LowImpact defaults or user overrides
$effectiveGpuThreshold = $null
$effectiveMemThreshold = $null
$effectiveCpuThreshold = $null
$effectiveSleepSeconds = $null

if ($PSBoundParameters.ContainsKey('GpuThreshold')) { $effectiveGpuThreshold = [int]$GpuThreshold }
elseif ($LowImpact) { $effectiveGpuThreshold = 40 }

if ($PSBoundParameters.ContainsKey('MemThreshold')) { $effectiveMemThreshold = [int]$MemThreshold }
elseif ($LowImpact) { $effectiveMemThreshold = 75 }

if ($PSBoundParameters.ContainsKey('CpuThreshold')) { $effectiveCpuThreshold = [int]$CpuThreshold }
elseif ($LowImpact) { $effectiveCpuThreshold = 70 }

if ($PSBoundParameters.ContainsKey('SleepSeconds')) { $effectiveSleepSeconds = [int]$SleepSeconds }
elseif ($LowImpact) { $effectiveSleepSeconds = 60 }

if ($effectiveGpuThreshold) { $processArgs += @("--gpu-threshold", $effectiveGpuThreshold) }
if ($effectiveMemThreshold) { $processArgs += @("--mem-threshold", $effectiveMemThreshold) }
if ($effectiveCpuThreshold) { $processArgs += @("--cpu-threshold", $effectiveCpuThreshold) }
if ($effectiveSleepSeconds) { $processArgs += @("--sleep-seconds", $effectiveSleepSeconds) }

if ($Verbose) {
    $processArgs += "--verbose"
    Write-Host "📺 Verbose mode: Starting in foreground (Ctrl+C to stop)" -ForegroundColor Yellow
    Write-Host "=" * 50 -ForegroundColor Gray
    
    # Run in foreground for real-time output
    & python $processArgs
} else {
    Write-Host "📝 Starting in background, logs: $logFile" -ForegroundColor Cyan
    
    # Start in background with logging
    $priority = if ($LowImpact) { 'BelowNormal' } else { 'Normal' }
    $process = Start-Process -FilePath "python" -ArgumentList $processArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput $logFile -RedirectStandardError "$logFile.err"
    # Set priority after launch (works in PS 5/7)
    if ($process) {
        try { $process.PriorityClass = $priority } catch { }
    }
    
    if ($process) {
        Write-Host " Orchestrator started (PID: $($process.Id))" -ForegroundColor Green
        Write-Host "📝 Monitor logs: Get-Content $logFile -Tail 10 -Wait" -ForegroundColor Cyan
        Write-Host "📊 Check status: .\setup_gpu_orchestrator.ps1 -Status" -ForegroundColor Cyan
    } else {
        Write-Host "X Failed to start orchestrator" -ForegroundColor Red
        exit 1
    }
}
