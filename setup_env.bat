@echo off
REM StreamSniped Environment Setup Script (Windows Batch)
REM This script loads all environment variables from config/streamsniped.env
REM Usage: setup_env.bat [options]

setlocal enabledelayedexpansion

REM Default values
set "ENV_FILE=config/streamsniped.env"
set "SHOW_VARS=false"
set "TEST_MODE=false"

REM Parse command line arguments
:parse_args
if "%1"=="" goto :main
if "%1"=="-Show" set "SHOW_VARS=true" & shift & goto :parse_args
if "%1"=="-Test" set "TEST_MODE=true" & shift & goto :parse_args
if "%1"=="-Help" goto :show_help
if "%1"=="-EnvFile" set "ENV_FILE=%2" & shift & shift & goto :parse_args
echo Unknown option: %1
goto :show_help

:show_help
echo StreamSniped Environment Setup Script
echo =====================================
echo.
echo Usage: setup_env.bat [options]
echo.
echo Options:
echo   -Show          Show all loaded environment variables
echo   -Test          Test environment setup without loading
echo   -Help          Show this help message
echo   -EnvFile       Specify custom env file path (default: config/streamsniped.env)
echo.
echo Examples:
echo   setup_env.bat                    # Load environment variables
echo   setup_env.bat -Show              # Show loaded variables
echo   setup_env.bat -Test              # Test without loading
echo.
echo After loading, you can run your StreamSniped commands:
echo   python docker-scripts/build_and_push_docker.py
echo   python aws-scripts/create_fargate_task_definition.py
echo   python aws-scripts/deploy_watcher.py
echo   python aws-scripts/trigger_manual_vod.py 2568791427
goto :end

:main
REM Check if env file exists
if not exist "%ENV_FILE%" (
    echo ❌ Environment file not found: %ENV_FILE%
    echo 💡 Make sure you're running this from the StreamSniped project root directory
    goto :end
)

echo 📁 Loading environment from: %ENV_FILE%

set "LOADED_COUNT=0"
set "SKIPPED_COUNT=0"

REM Load environment variables from file
for /f "usebackq tokens=1,2 delims==" %%a in ("%ENV_FILE%") do (
    set "line=%%a"
    set "key=%%a"
    set "value=%%b"
    
    REM Skip empty lines and comments
    if "!line!"=="" goto :next_line
    if "!line:~0,1!"=="#" goto :next_line
    
    REM Remove quotes if present
    if "!value:~0,1!"=="\"" (
        set "value=!value:~1!"
    )
    if "!value:~-1!"=="\"" (
        set "value=!value:~0,-1!"
    )
    
    REM Set environment variable
    set "!key!=!value!"
    set /a LOADED_COUNT+=1
    goto :next_line
    
    :next_line
)

echo ✅ Loaded %LOADED_COUNT% environment variables

if "%TEST_MODE%"=="true" (
    call :test_environment
    goto :end
)

echo 🚀 StreamSniped environment loaded successfully!
echo.
echo 💡 You can now run your StreamSniped commands:
echo    python docker-scripts/build_and_push_docker.py
echo    python aws-scripts/create_fargate_task_definition.py
echo    python aws-scripts/deploy_watcher.py
echo    python aws-scripts/trigger_manual_vod.py 2568791427
echo.
echo 🔍 Use -Show flag to see all loaded variables
echo 🧪 Use -Test flag to verify environment setup

if "%SHOW_VARS%"=="true" (
    call :show_environment
)

goto :end

:test_environment
echo 🧪 Testing environment setup...

set "REQUIRED_VARS=GPU_MODE GPU_ENABLED TWITCH_CLIENT_ID TWITCH_CLIENT_SECRET OPENROUTER_API_KEY AWS_REGION S3_BUCKET ECS_SUBNETS"
set "MISSING_COUNT=0"
set "PRESENT_COUNT=0"

for %%v in (%REQUIRED_VARS%) do (
    if defined %%v (
        set /a PRESENT_COUNT+=1
        echo    ✓ %%v
    ) else (
        set /a MISSING_COUNT+=1
        echo    ✗ %%v
    )
)

echo ✅ Present variables: %PRESENT_COUNT%
if %MISSING_COUNT% gtr 0 (
    echo ❌ Missing variables: %MISSING_COUNT%
    echo 🎉 All required environment variables are present!
) else (
    echo 🎉 All required environment variables are present!
)
goto :eof

:show_environment
echo.
echo 🔍 Current Environment Variables:
echo ==================================================
echo GPU_MODE=%GPU_MODE%
echo GPU_ENABLED=%GPU_ENABLED%
echo TWITCH_CLIENT_ID=%TWITCH_CLIENT_ID%
echo TWITCH_CLIENT_SECRET=***MASKED***
echo OPENROUTER_API_KEY=***MASKED***
echo AWS_REGION=%AWS_REGION%
echo S3_BUCKET=%S3_BUCKET%
echo ECS_SUBNETS=%ECS_SUBNETS%
echo ECS_SECURITY_GROUPS=%ECS_SECURITY_GROUPS%
echo RAG_ENABLED=%RAG_ENABLED%
echo CONTAINER_MODE=%CONTAINER_MODE%
echo PYTHONIOENCODING=%PYTHONIOENCODING%
goto :eof

:end
endlocal
