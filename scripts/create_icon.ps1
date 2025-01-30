# PowerShell script to create the icon and install required dependencies

# Check if pip is installed
if (!(Get-Command pip -ErrorAction SilentlyContinue)) {
    Write-Host "Error: pip is not installed. Please install Python and pip first."
    exit 1
}

# Install Pillow if not already installed
Write-Host "Installing/Updating Pillow library..."
python -m pip install Pillow

# Create the icon using our utility script
Write-Host "Creating icon..."
python ../utils/emoji_to_ico.py

# Verify the icon was created
if (Test-Path "../assets/icon.ico") {
    Write-Host "Icon created successfully at assets/icon.ico"
} else {
    Write-Host "Error: Failed to create icon"
    exit 1
}
