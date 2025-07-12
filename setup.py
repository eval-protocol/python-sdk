from setuptools import find_packages, setup

import versioneer

setup(
    name="reward-kit",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(include=["reward_kit*", "reward_protocol*", "development*"]),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=2.0.0",
        "dataclasses-json>=0.5.7",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "openai==1.78.1",
        "aiosqlite",
        "aiohttp",
        "mcp>=1.9.2",
        "PyYAML>=5.0",  # Added PyYAML
        "datasets==3.6.0",  # For dataset loading functionality
        "fsspec==2025.3.0",  # For filesystem interactions, pinned for dataset compatibility
        "hydra-core>=1.3.2",  # For configuration management
        "omegaconf>=2.3.0",  # For configuration objects
        "gymnasium>=0.29.0",  # For FrozenLake and other gym environments
        "httpx>=0.24.0",  # For HTTP client functionality in agent resources
        "fireworks-ai>=0.17.19",  # For Fireworks Build SDK integration
    ],
    extras_require={
        "dev": [
            "build",
            "twine",
            "pytest>=6.0.0",
            "pytest-asyncio",
            "pytest-httpserver",
            "werkzeug>=2.0.0",  # needed for test_url_handling.py
            "black>=21.5b2",
            "mypy>=0.812",
            "flake8>=3.9.2",
            "autopep8>=1.5.0",
            "transformers>=4.0.0",
            "types-setuptools",
            "types-requests",
            "types-PyYAML",
            "types-docker",
            "versioneer>=0.20",
            "openai==1.78.1",  # needed for tests using OpenAI types
            # datasets, hydra-core, omegaconf moved to core dependencies
            "pre-commit",
            "e2b>=0.15.0",  # Added e2b for E2B environment tests
            "docker==7.1.0",
        ],
        "trl": [
            "torch>=1.9",
            "trl>=0.7.0",
            "peft>=0.7.0",
            "transformers>=4.0.0",
            "accelerate>=0.28.0",
        ],
        "deepseek": [
            "difflib>=3.0.0",
        ],
        # "deepeval": ["deepeval>=3.1.0"],
        "openevals": ["openevals>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "fireworks-reward=reward_kit.cli:main",
            "reward-kit=reward_kit.cli:main",
            "reward-protocol=reward_kit.cli:main",
        ],
    },
    author="Fireworks AI",
    author_email="info@fireworks.ai",
    description="A Python library for defining, testing, and using reward functions",
    long_description="A library for defining, testing, and deploying reward functions",
    url="https://github.com/fireworks-ai/reward-kit",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
