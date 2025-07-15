#!/bin/zsh
# Add UV to your PATH permanently

echo "Adding UV to your PATH in .zshrc..."

# Check if UV path is already in .zshrc
if ! grep -q "/.local/bin" ~/.zshrc; then
    echo "" >> ~/.zshrc
    echo "# UV package manager" >> ~/.zshrc
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
    echo "Added UV to PATH in .zshrc"
else
    echo "UV path already in .zshrc"
fi

echo ""
echo "To apply changes now, run:"
echo "  source ~/.zshrc"
echo ""
echo "UV is installed at: $HOME/.local/bin/uv"
echo ""
echo "For this project, activate the virtual environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then you can use UV to manage packages:"
echo "  uv pip install <package>"
echo "  uv pip list"
echo ""
echo "Regarding conda: Your conda installation needs repair."
echo "You can either:"
echo "1. Continue using UV (recommended for this project)"
echo "2. Reinstall miniconda later (your environments are preserved)"