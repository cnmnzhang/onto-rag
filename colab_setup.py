"""
Google Colab Setup and Environment Detection
Run this first in Colab to check GPU/RAM and install dependencies
"""

import os
import sys


def detect_colab_environment():
    """Detect if running in Google Colab and check resources."""
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if not in_colab:
        print("Not running in Google Colab")
        return False

    print("=" * 60)
    print("Google Colab Environment Detected")
    print("=" * 60)

    # Check GPU
    print("\nGPU Information:")
    try:
        gpu_info = get_ipython().getoutput('nvidia-smi')
        gpu_info_str = '\n'.join(gpu_info)
        if gpu_info_str.find('failed') >= 0:
            print('✗ Not connected to a GPU')
            print('  To enable GPU: Runtime > Change runtime type > Hardware accelerator > GPU')
        else:
            # Extract GPU name
            for line in gpu_info:
                if 'Tesla' in line or 'T4' in line or 'P100' in line or 'V100' in line or 'A100' in line:
                    print(f'✓ GPU detected: {line.strip()}')
                    break
            else:
                print('✓ GPU available')
    except:
        print('✗ Could not detect GPU')

    # Check RAM
    print("\nRAM Information:")
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f'✓ Available RAM: {ram_gb:.1f} GB')

        if ram_gb < 12:
            print('  Standard RAM runtime')
        elif ram_gb < 20:
            print('  High-RAM runtime')
        else:
            print('  Premium high-RAM runtime')
    except ImportError:
        print('✗ psutil not installed (will be installed with requirements)')

    print("\n" + "=" * 60)
    return True


def install_dependencies():
    """Install required packages for Colab."""
    print("\n" + "=" * 60)
    print("Installing Dependencies")
    print("=" * 60)

    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("\n✗ requirements.txt not found")
        print("  Creating basic requirements...")

        requirements = """pandas
requests
numpy
scikit-learn
sentence-transformers
google-generativeai
psutil
"""
        with open('requirements.txt', 'w') as f:
            f.write(requirements)

    print("\nInstalling packages from requirements.txt...")
    os.system('pip install -q -r requirements.txt')
    print("✓ Dependencies installed")


def setup_environment_variables():
    """Setup environment variables for Colab."""
    print("\n" + "=" * 60)
    print("Environment Setup")
    print("=" * 60)

    # Check for API keys
    google_key = os.getenv("GOOGLE_API_KEY")
    bioportal_key = os.getenv("BIOPORTAL_API_KEY")

    if not google_key:
        print("\n⚠ GOOGLE_API_KEY not set")
        print("  Get a free API key from: https://aistudio.google.com/app/apikey")
        print("  Then run: os.environ['GOOGLE_API_KEY'] = 'your-key-here'")
    else:
        print("✓ GOOGLE_API_KEY is set")

    if not bioportal_key:
        print("\n⚠ BIOPORTAL_API_KEY not set")
        print("  Set with: os.environ['BIOPORTAL_API_KEY'] = 'your-key-here'")
    else:
        print("✓ BIOPORTAL_API_KEY is set")

    # Default to Gemini in Colab (free)
    if not os.getenv("USE_HUGGINGFACE"):
        print("\n✓ Defaulting to Google Gemini (free API)")
        print("  To use Hugging Face instead: os.environ['USE_HUGGINGFACE'] = 'true'")


def run_colab_setup():
    """Run complete Colab setup."""
    is_colab = detect_colab_environment()

    if is_colab:
        install_dependencies()
        setup_environment_variables()

        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Set your GOOGLE_API_KEY if not already set")
        print("2. Set your BIOPORTAL_API_KEY")
        print("3. Run the main notebook cells")
        print("\n" + "=" * 60)
    else:
        print("\nRunning locally. Use:")
        print("  pip install -r requirements.txt")
        print("  export GOOGLE_API_KEY='your-key'")
        print("  export BIOPORTAL_API_KEY='your-key'")


if __name__ == "__main__":
    run_colab_setup()
