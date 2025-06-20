#!/bin/bash

# Setup GitHub Secrets for CI/CD Pipeline
# Usage: ./scripts/setup-github-secrets.sh

echo "🔐 BEM System GitHub Secrets Setup"
echo "================================="
echo ""
echo "This script will help you set up the required GitHub secrets for the CI/CD pipeline."
echo "You'll need the GitHub CLI (gh) installed and authenticated."
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) not found. Please install it first:"
    echo "   brew install gh"
    echo "   gh auth login"
    exit 1
fi

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "📦 Repository: $REPO"
echo ""

# Function to set secret
set_secret() {
    local SECRET_NAME=$1
    local SECRET_VALUE=$2
    local DESCRIPTION=$3
    
    echo "Setting $SECRET_NAME: $DESCRIPTION"
    echo "$SECRET_VALUE" | gh secret set "$SECRET_NAME" --repo "$REPO"
}

echo "📝 Please provide the following values:"
echo ""

# Render API Key
read -p "1. Render API Key (from https://dashboard.render.com/account/api-keys): " RENDER_API_KEY
set_secret "RENDER_API_KEY" "$RENDER_API_KEY" "Render deployment API key"

# Render Service IDs
echo ""
echo "To find Service IDs, go to each service in Render dashboard and look at the URL."
echo "Example: https://dashboard.render.com/web/srv-XXXXX <- 'srv-XXXXX' is the service ID"
echo ""
read -p "2. Render AA Service ID (srv-xxxxx): " RENDER_SERVICE_ID_AA
set_secret "RENDER_SERVICE_ID_AA" "$RENDER_SERVICE_ID_AA" "AA service ID in Render"

read -p "3. Render ECM Service ID (srv-xxxxx): " RENDER_SERVICE_ID_ECM
set_secret "RENDER_SERVICE_ID_ECM" "$RENDER_SERVICE_ID_ECM" "ECM service ID in Render"

# Database URL
echo ""
echo "Database URL format: postgresql://user:password@host:port/database?sslmode=require"
read -p "4. Production Database URL: " DATABASE_URL
set_secret "DATABASE_URL" "$DATABASE_URL" "PostgreSQL connection string"

# Optional: Slack Webhook
echo ""
read -p "5. Slack Webhook URL (optional, press Enter to skip): " SLACK_WEBHOOK_URL
if [ ! -z "$SLACK_WEBHOOK_URL" ]; then
    set_secret "SLACK_WEBHOOK_URL" "$SLACK_WEBHOOK_URL" "Slack notification webhook"
fi

# Optional: Twilio for OTP
echo ""
echo "For SMS OTP authentication (optional):"
read -p "6. Twilio Account SID (optional): " TWILIO_ACCOUNT_SID
if [ ! -z "$TWILIO_ACCOUNT_SID" ]; then
    set_secret "TWILIO_ACCOUNT_SID" "$TWILIO_ACCOUNT_SID" "Twilio account SID"
    
    read -p "7. Twilio Auth Token: " TWILIO_AUTH_TOKEN
    set_secret "TWILIO_AUTH_TOKEN" "$TWILIO_AUTH_TOKEN" "Twilio auth token"
    
    read -p "8. Twilio From Number (+1234567890): " TWILIO_FROM_NUMBER
    set_secret "TWILIO_FROM_NUMBER" "$TWILIO_FROM_NUMBER" "Twilio sender number"
fi

# Optional: AWS for backups
echo ""
echo "For AWS S3 backup storage (optional):"
read -p "9. AWS Access Key ID (optional): " AWS_ACCESS_KEY_ID
if [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
    set_secret "AWS_ACCESS_KEY_ID" "$AWS_ACCESS_KEY_ID" "AWS access key"
    
    read -s -p "10. AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    echo ""
    set_secret "AWS_SECRET_ACCESS_KEY" "$AWS_SECRET_ACCESS_KEY" "AWS secret key"
    
    read -p "11. AWS Region (e.g., us-east-1): " AWS_REGION
    set_secret "AWS_REGION" "$AWS_REGION" "AWS region"
    
    read -p "12. AWS S3 Bucket Name: " AWS_BACKUP_BUCKET
    set_secret "AWS_BACKUP_BUCKET" "$AWS_BACKUP_BUCKET" "S3 bucket for backups"
fi

echo ""
echo "✅ GitHub secrets setup complete!"
echo ""
echo "📋 Summary of configured secrets:"
gh secret list --repo "$REPO"

echo ""
echo "🚀 Next steps:"
echo "1. Commit and push your code to trigger the CI/CD pipeline"
echo "2. Monitor the Actions tab in GitHub for deployment progress"
echo "3. Check Render dashboard for service status"
echo ""
echo "To trigger DGL retraining, include '[retrain]' in your commit message" 