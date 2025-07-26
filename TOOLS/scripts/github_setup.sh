#!/bin/bash

# GitHub Setup Script for Fungal Electrical Activity Analysis
# This script helps authenticate with GitHub and push the repository

echo "🍄 Setting up GitHub for Fungal Electrical Activity Analysis..."
echo "================================================================"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run 'git init' first."
    exit 1
fi

# Check if we have commits
if ! git rev-parse HEAD >/dev/null 2>&1; then
    echo "❌ No commits found. Please commit your changes first."
    exit 1
fi

echo "✅ Repository is ready!"
echo ""
echo "🔐 GitHub Authentication Options:"
echo ""
echo "Option 1: GitHub CLI (Recommended)"
echo "-----------------------------------"
echo "1. Run: gh auth login"
echo "2. Follow the prompts to authenticate"
echo "3. Then run: ./push_repository.sh"
echo ""
echo "Option 2: Personal Access Token"
echo "-------------------------------"
echo "1. Go to https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Give it a name like 'Fungal Analysis Project'"
echo "4. Select scopes: repo, workflow"
echo "5. Copy the token"
echo "6. Run: export GH_TOKEN=your_token_here"
echo "7. Then run: ./push_repository.sh"
echo ""
echo "Option 3: Manual Setup"
echo "----------------------"
echo "1. Go to https://github.com/new"
echo "2. Create repository: fungal-electrical-analysis"
echo "3. Make it public"
echo "4. Don't initialize with README"
echo "5. Copy the repository URL"
echo "6. Run the commands shown below"
echo ""

echo "📋 Manual Commands (if you choose Option 3):"
echo "git remote add origin https://github.com/YOUR_USERNAME/fungal-electrical-analysis.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""

echo "🎯 Repository Details:"
echo "- Name: fungal-electrical-analysis"
echo "- Description: Advanced wave transform analysis for fungal electrical activity"
echo "- Visibility: Public"
echo "- Topics: fungal-electrical-activity, wave-transform-analysis, adamatzky-research"
echo ""

echo "📊 What will be pushed:"
echo "- 60 files with 16,339 insertions"
echo "- Main analysis script (3,558 lines)"
echo "- Comprehensive documentation"
echo "- Biological validation report"
echo "- MIT license with scientific attribution"
echo ""

echo "🚀 Ready to push your fungal electrical activity research to GitHub!"
echo ""
echo "💡 Pro tip: After pushing, consider:"
echo "   - Adding detailed repository description"
echo "   - Creating issues for future enhancements"
echo "   - Setting up GitHub Actions for testing"
echo "   - Adding collaborators if working with a team"
echo ""
echo "🔬 Scientific Impact:"
echo "- Enables cross-species analysis of fungal electrical communication"
echo "- Provides data-driven methodology for biological signal processing"
echo "- Supports Adamatzky's research with computational validation"
echo "- Discovers new datasets for fungal electrical activity research" 