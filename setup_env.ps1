# StreamSniped Environment Setup Script
# This script loads all environment variables from config/streamsniped.env
# Usage: .\setup_env.ps1 [options]

param(
    [switch]$Show,
    [switch]$Test,
    [switch]$Help,
    [string]$EnvFile = "config/streamsniped.env"
)

function Show-Help {
    Write-Host "StreamSniped Environment Setup Script" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\setup_env.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Show          Show all loaded environment variables" -ForegroundColor White
    Write-Host "  -Test          Test environment setup without loading" -ForegroundColor White
    Write-Host "  -Help          Show this help message" -ForegroundColor White
    Write-Host "  -EnvFile       Specify custom env file path (default: config/streamsniped.env)" -ForegroundColor White
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\setup_env.ps1                    # Load environment variables" -ForegroundColor White
    Write-Host "  .\setup_env.ps1 -Show              # Show loaded variables" -ForegroundColor White
    Write-Host "  .\setup_env.ps1 -Test              # Test without loading" -ForegroundColor White
    Write-Host ""
    Write-Host "After loading, you can run your StreamSniped commands:" -ForegroundColor Green
    Write-Host "  python docker-scripts/build_and_push_docker.py" -ForegroundColor White
    Write-Host "  python aws-scripts/create_fargate_task_definition.py" -ForegroundColor White
    Write-Host "  python aws-scripts/deploy_watcher.py" -ForegroundColor White
    Write-Host "  python aws-scripts/trigger_manual_vod.py 2568791427" -ForegroundColor White
}

function Load-EnvironmentFile {
    param([string]$FilePath)
    
    if (-not (Test-Path $FilePath)) {
        Write-Host "❌ Environment file not found: $FilePath" -ForegroundColor Red
        Write-Host "💡 Make sure you're running this from the StreamSniped project root directory" -ForegroundColor Yellow
        return $false
    }
    
    $loadedCount = 0
    $skippedCount = 0
    
    Write-Host "📁 Loading environment from: $FilePath" -ForegroundColor Cyan
    
    Get-Content $FilePath | ForEach-Object {
        $line = $_.Trim()
        
        # Skip empty lines and comments
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
            return
        }
        
        # Parse KEY=VALUE format
        if ($line -match "^([^=]+)=(.*)$") {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove quotes if present
            if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            
            # Set environment variable
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
            $loadedCount++
        } else {
            $skippedCount++
        }
    }
    
    Write-Host "✅ Loaded $loadedCount environment variables" -ForegroundColor Green
    if ($skippedCount -gt 0) {
        Write-Host "⚠️  Skipped $skippedCount lines (comments or invalid format)" -ForegroundColor Yellow
    }
    
    return $true
}

function Show-EnvironmentVariables {
    Write-Host "`n🔍 Current Environment Variables:" -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Gray
    
    # Get all environment variables and filter for StreamSniped related ones
    $envVars = Get-ChildItem Env: | Where-Object {
        $_.Name -match "^(GPU_|TWITCH_|YOUTUBE_|OPENAI_|OPENROUTER_|GEMINI_|AWS_|ECS_|S3_|DYNAMODB_|RAG_|CONTAINER_|PYTHON_|WHISPER_|FFMPEG_|MAX_|MIN_|LEAD_|TARGET_|FOCUSED_|TEST_|ENABLE_|UPLOAD_|QUALITY_|OFFLOAD_|USE_|SKIP_|RENDER_|LOG_|TEMP_|STORAGE_|ENVIRONMENT_|CUDA_|NVIDIA_|FULL_QUEUE_URL|DOWNLOAD_MAX_CONCURRENCY|TD_MAX_PARALLEL)"
    } | Sort-Object Name
    
    if ($envVars.Count -eq 0) {
        Write-Host "No StreamSniped environment variables found" -ForegroundColor Yellow
        return
    }
    
    foreach ($var in $envVars) {
        $value = $var.Value
        # Mask sensitive values
        if ($var.Name -match "(SECRET|KEY|TOKEN|PASSWORD)") {
            $value = "***MASKED***"
        }
        Write-Host "$($var.Name) = $value" -ForegroundColor White
    }
}

function Test-EnvironmentSetup {
    Write-Host "🧪 Testing environment setup..." -ForegroundColor Cyan
    
    $requiredVars = @(
        "GPU_MODE", "GPU_ENABLED", "TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET",
        "OPENROUTER_API_KEY", "AWS_REGION", "S3_BUCKET", "ECS_SUBNETS"
    )
    
    $missingVars = @()
    $presentVars = @()
    
    foreach ($var in $requiredVars) {
        $value = [Environment]::GetEnvironmentVariable($var, "Process")
        if ([string]::IsNullOrEmpty($value)) {
            $missingVars += $var
        } else {
            $presentVars += $var
        }
    }
    
    Write-Host "✅ Present variables: $($presentVars.Count)/$($requiredVars.Count)" -ForegroundColor Green
    foreach ($var in $presentVars) {
        Write-Host "   ✓ $var" -ForegroundColor Green
    }
    
    if ($missingVars.Count -gt 0) {
        Write-Host "❌ Missing variables: $($missingVars.Count)" -ForegroundColor Red
        foreach ($var in $missingVars) {
            Write-Host "   ✗ $var" -ForegroundColor Red
        }
        return $false
    }
    
    Write-Host "🎉 All required environment variables are present!" -ForegroundColor Green
    return $true
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

if ($Test) {
    if (Load-EnvironmentFile -FilePath $EnvFile) {
        Test-EnvironmentSetup
    }
    exit 0
}

# Load environment variables
if (Load-EnvironmentFile -FilePath $EnvFile) {
    Write-Host "🚀 StreamSniped environment loaded successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 You can now run your StreamSniped commands:" -ForegroundColor Cyan
    Write-Host "   python docker-scripts/build_and_push_docker.py" -ForegroundColor White
    Write-Host "   python aws-scripts/create_fargate_task_definition.py" -ForegroundColor White
    Write-Host "   python aws-scripts/deploy_watcher.py" -ForegroundColor White
    Write-Host "   python aws-scripts/trigger_manual_vod.py 2568791427" -ForegroundColor White
    Write-Host ""
    Write-Host "🔍 Use -Show flag to see all loaded variables" -ForegroundColor Yellow
    Write-Host "🧪 Use -Test flag to verify environment setup" -ForegroundColor Yellow
}

if ($Show) {
    Show-EnvironmentVariables
}
