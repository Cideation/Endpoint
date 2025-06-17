#!/bin/bash

# Install Railway CLI if not installed
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm i -g @railway/cli
fi

# Login to Railway (if not already logged in)
railway login

# Link to your project
railway link

# Set environment variables
railway variables set FLASK_ENV=production

# Deploy the project
railway up

echo "Deployment complete! Check your Railway dashboard for the URL." 