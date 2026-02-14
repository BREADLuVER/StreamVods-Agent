# Setup environment variables for clip generation offloading
# Run this script before starting the clip daemon

Write-Host "Setting up clip generation offload environment..." -ForegroundColor Green

# Clip generation offload settings
$env:SKIP_CLIP_ENCODING_IN_CLOUD = "true"
$env:OFFLOAD_CLIPS = "true"
$env:CLIP_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/590184039189/streamsniped-clip-queue"

# Advanced layout system
$env:CLIP_LAYOUT_SYSTEM = "advanced"

# Quality and upload settings
$env:QUALITY = "1080p"
$env:UPLOAD_YOUTUBE = "true"

# AWS settings
$env:AWS_REGION = "us-east-1"
$env:S3_BUCKET = "streamsniped-dev-videos"

Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "  SKIP_CLIP_ENCODING_IN_CLOUD: $env:SKIP_CLIP_ENCODING_IN_CLOUD"
Write-Host "  OFFLOAD_CLIPS: $env:OFFLOAD_CLIPS"
Write-Host "  CLIP_QUEUE_URL: $env:CLIP_QUEUE_URL"
Write-Host "  CLIP_LAYOUT_SYSTEM: $env:CLIP_LAYOUT_SYSTEM"
Write-Host "  QUALITY: $env:QUALITY"
Write-Host "  UPLOAD_YOUTUBE: $env:UPLOAD_YOUTUBE"

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Ensure SQS queue is created: python aws-scripts/setup_clip_queue.py"
Write-Host "2. Start clip daemon: python aws-scripts/clip_daemon.py --queue-url $env:CLIP_QUEUE_URL"
Write-Host "3. Run cloud workflow with clip offloading enabled"

Write-Host "`nEnvironment ready for clip generation offloading!" -ForegroundColor Green
