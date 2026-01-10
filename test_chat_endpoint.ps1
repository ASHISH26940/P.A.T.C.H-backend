# Test script for P.A.T.C.H backend chat endpoint
# This script tests authentication and chat endpoint

Write-Host "`n=== Testing P.A.T.C.H Backend Authentication & Chat ===" -ForegroundColor Cyan

# Step 1: Login and get JWT token
Write-Host "`n[1/3] Logging in to get JWT token..." -ForegroundColor Yellow

# Replace these with your actual credentials
$username = "Ashish ashish"  # Based on your logs
$password = "your_password_here"  # Replace with actual password

$loginBody = @{
    username = $username
    password = $password
} | ConvertTo-Json

try {
    $loginResponse = Invoke-RestMethod -Uri "http://127.0.0.1:5000/v1/auth/login" `
        -Method Post `
        -ContentType "application/x-www-form-urlencoded" `
        -Body "username=$username&password=$password"
    
    $token = $loginResponse.access_token
    Write-Host "✓ Login successful!" -ForegroundColor Green
    Write-Host "Token (first 20 chars): $($token.Substring(0,20))..." -ForegroundColor Gray
} catch {
    Write-Host "✗ Login failed: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Test /v1/auth/me endpoint (diagnostic check)
Write-Host "`n[2/3] Testing /v1/auth/me endpoint..." -ForegroundColor Yellow

$headers = @{
    "Authorization" = "Bearer $token"
}

try {
    $meResponse = Invoke-RestMethod -Uri "http://127.0.0.1:5000/v1/auth/me" `
        -Method Get `
        -Headers $headers
    
    Write-Host "✓ /v1/auth/me successful!" -ForegroundColor Green
    Write-Host "User: $($meResponse.username)" -ForegroundColor Gray
} catch {
    Write-Host "✗ /v1/auth/me failed: $_" -ForegroundColor Red
    Write-Host $_.Exception.Response.StatusCode -ForegroundColor Red
}

# Step 3: Test /v1/chat/ endpoint
Write-Host "`n[3/3] Testing /v1/chat/ endpoint..." -ForegroundColor Yellow

$chatBody = @{
    user_id = "2"
    message = "Hello, this is a test message"
    conversation_id = "test-conversation-123"
    collection_name = "test-collection"
} | ConvertTo-Json

$chatHeaders = @{
    "Authorization" = "Bearer $token"
    "Content-Type" = "application/json"
}

try {
    Write-Host "Sending POST request to /v1/chat/..." -ForegroundColor Gray
    $chatResponse = Invoke-RestMethod -Uri "http://127.0.0.1:5000/v1/chat/" `
        -Method Post `
        -Headers $chatHeaders `
        -Body $chatBody
    
    Write-Host "✓ /v1/chat/ successful!" -ForegroundColor Green
    Write-Host "Response: $($chatResponse | ConvertTo-Json -Depth 3)" -ForegroundColor Gray
} catch {
    Write-Host "✗ /v1/chat/ failed!" -ForegroundColor Red
    Write-Host "Status Code: $($_.Exception.Response.StatusCode.value__)" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    
    # Try to get error details
    try {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $errorBody = $reader.ReadToEnd()
        Write-Host "Error details: $errorBody" -ForegroundColor Red
    } catch {
        Write-Host "Could not read error response body" -ForegroundColor Red
    }
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
