#!/bin/zsh
# Script to fix conda installation

echo "Fixing conda installation..."

# Remove the broken conda initialization from .zshrc
echo "Backing up .zshrc to .zshrc.backup"
cp ~/.zshrc ~/.zshrc.backup

# Remove old conda init block
sed -i '' '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.zshrc

# Since conda executable is missing, we'll use micromamba or reinstall
echo "Your conda installation is corrupted. You have two options:"
echo ""
echo "Option 1: Install micromamba (recommended, compatible with conda envs)"
echo "  Run: curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba"
echo "  Then: ./bin/micromamba shell init -s zsh -p ~/micromamba"
echo ""
echo "Option 2: Reinstall miniconda (will preserve environments)"
echo "  Download from: https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
echo "  Run: bash Miniconda3-latest-MacOSX-arm64.sh -b -u -p ~/miniconda3"
echo ""
echo "Your existing environments are preserved in: /Users/rachael/miniconda3/envs/"
echo ""
echo "For now, let's focus on getting UV working for this project."