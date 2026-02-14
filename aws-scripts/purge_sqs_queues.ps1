Param(
    [string[]]$QueueUrls,
    [switch]$IncludeYouTube,
    [int]$WaitSeconds = 65,
    [string]$Region
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err($msg) { Write-Host $msg -ForegroundColor Red }

function Test-AwsCli {
    try {
        $null = aws --version
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-AwsCli)) {
    Write-Err "AWS CLI not found. Install from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
}

# Build queue list from args or environment
$queues = @()
if ($QueueUrls -and $QueueUrls.Count -gt 0) {
    $queues = $QueueUrls
} else {
    $envQueues = @(
        $env:FULL_QUEUE_URL,
        $env:CLIP_QUEUE_URL,
        $env:RENDER_QUEUE_URL
    )
    foreach ($q in $envQueues) {
        if ($q -and $q.Trim() -ne '') { $queues += $q }
    }
    if ($IncludeYouTube) {
        if ($env:YOUTUBE_UPLOAD_QUEUE_URL -and $env:YOUTUBE_UPLOAD_QUEUE_URL.Trim() -ne '') {
            $queues += $env:YOUTUBE_UPLOAD_QUEUE_URL
        }
    }
}

if ($queues.Count -eq 0) {
    Write-Err "No queues provided or found in environment. Set FULL_QUEUE_URL, CLIP_QUEUE_URL, RENDER_QUEUE_URL (and optionally YOUTUBE_UPLOAD_QUEUE_URL) or pass -QueueUrls."
    exit 1
}

Write-Info "Purging the following queues:";
$queues | ForEach-Object { Write-Host " - $_" }

Write-Warn "Ensure your orchestrator is stopped to avoid immediate re-enqueues."

function Invoke-Aws([string]$service, [string[]]$args) {
    if ($Region -and $Region.Trim() -ne '') {
        aws $service @args --region $Region
    } else {
        aws $service @args
    }
}

foreach ($q in $queues) {
    try {
        Write-Info "Purging $q ..."
        Invoke-Aws sqs @('purge-queue', '--queue-url', $q) | Out-Null
        Write-Host "  -> purge requested"
    } catch {
        Write-Warn "  -> purge request failed: $($_.Exception.Message)"
    }
}

Write-Info "Waiting $WaitSeconds seconds for in-flight messages to clear (SQS purge throttle is 60s)..."
Start-Sleep -Seconds $WaitSeconds

Write-Info "Queue attributes after purge:"
foreach ($q in $queues) {
    try {
        Invoke-Aws sqs @('get-queue-attributes', '--queue-url', $q, '--attribute-names', 'ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible')
    } catch {
        Write-Warn "  -> failed to fetch attributes for $q`: $($_.Exception.Message)"
    }
}

Write-Info "Done."


