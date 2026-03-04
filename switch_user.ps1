param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("saleha", "asma")]
    [string]$user = ""
)

$repo_owner = "asmasaeed18as"
$repo_name = "Nust-Banking-Customer-Service-AI-Agent"

# Define User Profiles
$profiles = @{
    "saleha" = @{
        "name"  = "saleha-zf"
        "email" = "saleha.zf22@gmail.com"
        "github_user" = "saleha-zf"
    }
    "asma" = @{
        "name"  = "asmasaeed18as"
        "email" = "asmasaeed18as@users.noreply.github.com"
        "github_user" = "asmasaeed18as"
    }
}

# If no user provided, show current and ask
if ($user -eq "") {
    $current_name = $(git config user.name)
    Write-Host "Current Git User: $current_name" -ForegroundColor Cyan
    $choice = Read-Host "Switch to (saleha/asma)?"
    $user = $choice.ToLower()
}

if ($profiles.ContainsKey($user)) {
    $p = $profiles[$user]
    
    # 1. Update Git Config
    git config user.name $p.name
    git config user.email $p.email
    
    # 2. Update Remote URL for Push Authentication
    $new_url = "https://$($p.github_user)@github.com/$repo_owner/$repo_name.git"
    git remote set-url origin $new_url
    
    Write-Host "`n----------------------------------------" -ForegroundColor Green
    Write-Host "✅ SUCCESSFULLY SWITCHED TO: $($p.name)" -ForegroundColor Green
    Write-Host "📧 Email: $($p.email)"
    Write-Host "🔗 Local Remote URL updated for authentication."
    Write-Host "----------------------------------------`n"
} else {
    Write-Host "❌ Invalid user. Please choose 'saleha' or 'asma'." -ForegroundColor Red
}
