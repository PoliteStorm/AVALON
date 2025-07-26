#!/bin/bash

echo "🚀 Pushing Fungal Electrical Activity Analysis to GitHub"
echo "========================================================"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository."
    exit 1
fi

# Check if we have commits
if ! git rev-parse HEAD >/dev/null 2>&1; then
    echo "❌ No commits found."
    exit 1
fi

echo "✅ Repository is ready!"
echo ""

# Check if remote origin exists
if git remote get-url origin >/dev/null 2>&1; then
    REMOTE_URL=$(git remote get-url origin)
    echo "📡 Remote URL: $REMOTE_URL"
    echo ""
    echo "🔄 Pushing to GitHub..."
    echo ""
    
    # Rename branch to main
    echo "📝 Renaming branch to main..."
    git branch -M main
    
    # Push to GitHub
    echo "📤 Pushing to GitHub..."
    if git push -u origin main; then
        echo ""
        echo "🎉 SUCCESS! Repository pushed to GitHub!"
        echo ""
        echo "📊 Repository Statistics:"
        echo "- Files pushed: 60 files, 16,339 insertions"
        echo "- Main analysis script: ultra_simple_scaling_analysis.py (3,558 lines)"
        echo "- Biological review: BIOLOGICAL_REVIEW_REPORT.md"
        echo "- Project documentation: README.md"
        echo "- Requirements: requirements.txt"
        echo "- License: MIT with scientific attribution"
        echo ""
        echo "🌐 Your repository is now live on GitHub!"
        echo "Visit: $REMOTE_URL"
        echo ""
        echo "💡 Next steps:"
        echo "1. Add repository topics: fungal-electrical-activity, wave-transform-analysis, adamatzky-research"
        echo "2. Create a release: v1.0.0"
        echo "3. Add collaborators if working with a team"
        echo "4. Set up GitHub Actions for automated testing"
        echo ""
        echo "🎉 Congratulations on publishing your fungal electrical activity research!"
    else
        echo ""
        echo "❌ Failed to push to GitHub."
        echo ""
        echo "🔧 Troubleshooting:"
        echo "1. Check your GitHub authentication"
        echo "2. Verify the remote URL: git remote -v"
        echo "3. Make sure the repository exists on GitHub"
        echo ""
    fi
else
    echo "❌ No remote origin configured."
    echo ""
    echo "🔧 Please set up the remote origin:"
    echo ""
    echo "1. First, create the repository on GitHub:"
    echo "   - Go to https://github.com/new"
    echo "   - Name: fungal-electrical-analysis"
    echo "   - Description: Advanced wave transform analysis for fungal electrical activity"
    echo "   - Make it public"
    echo "   - Don't initialize with README"
    echo ""
    echo "2. Then add the remote origin:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/fungal-electrical-analysis.git"
    echo ""
    echo "3. Then run this script again:"
    echo "   ./push_to_github.sh"
    echo ""
fi 