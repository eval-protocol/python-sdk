# Cache for PEP 440 version string
import subprocess

from typing import Dict, Optional, TypedDict


class _VersionCache(TypedDict):
    version: Optional[str]
    base_version: Optional[str]


_version_cache: _VersionCache = {"version": None, "base_version": None}


def get_pep440_version(base_version=None):
    """
    Generate a PEP 440 compliant version string based on git information.

    This function is inspired by versioneer but doesn't require the full versioneer
    setup, making it easier for downstream users to adopt without additional dependencies.

    The result is cached statically to avoid repeated git calls.

    Args:
        base_version: The base version string (e.g., "1.0.0"). If None, will try to
                     find the most recent version tag in git.

    Returns:
        A PEP 440 compliant version string that includes:
        - Development release number (devN) based on commit count since base_version
        - Local version identifier with git commit hash
        - Dirty indicator if there are uncommitted changes

    Examples:
        >>> get_pep440_version("1.0.0")
        "1.0.0.dev42+g1234567"  # 42 commits since 1.0.0, commit hash 1234567
        >>> get_pep440_version("1.0.0")  # with uncommitted changes
        "1.0.0.dev42+g1234567.dirty"  # indicates dirty working directory
        >>> get_pep440_version("1.0.0")  # no git available
        "1.0.0+unknown"  # indicates git info not available
    """
    # Check if we have a cached version for this base_version
    if _version_cache["version"] is not None and _version_cache["base_version"] == base_version:
        return _version_cache["version"]
    try:
        # Check if we're in a git repository
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        # If base_version is None, try to find the most recent version tag
        if base_version is None:
            try:
                base_version = subprocess.check_output(
                    ["git", "describe", "--tags", "--abbrev=0"], universal_newlines=True, stderr=subprocess.DEVNULL
                ).strip()
            except subprocess.CalledProcessError:
                # No tags found, we'll handle this case specially
                base_version = None

        # Get commit count since base_version
        if base_version is None:
            # No base version (no tags), just count all commits
            count = subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"], universal_newlines=True, stderr=subprocess.DEVNULL
            ).strip()
            base_version = "0.0.0"  # Use this for the final version string
        else:
            try:
                count = subprocess.check_output(
                    ["git", "rev-list", "--count", f"{base_version}..HEAD"],
                    universal_newlines=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                # If no commits found, try counting from the beginning
                if count == "0" or not count:
                    count = subprocess.check_output(
                        ["git", "rev-list", "--count", "HEAD"], universal_newlines=True, stderr=subprocess.DEVNULL
                    ).strip()
            except subprocess.CalledProcessError:
                # If base_version tag doesn't exist, count all commits
                count = subprocess.check_output(
                    ["git", "rev-list", "--count", "HEAD"], universal_newlines=True, stderr=subprocess.DEVNULL
                ).strip()

        # Get short commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], universal_newlines=True, stderr=subprocess.DEVNULL
        ).strip()

        # Check for uncommitted changes (dirty working directory)
        try:
            subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            dirty_suffix = ""
        except subprocess.CalledProcessError:
            dirty_suffix = ".dirty"

        # Ensure count is a valid integer
        try:
            dev_count = int(count)
        except (ValueError, TypeError):
            dev_count = 0

        # Build PEP 440 compliant version string
        # Format: <base_version>.dev<count>+g<hash>[.dirty]
        version_parts = [base_version]

        if dev_count > 0:
            version_parts.append(f".dev{dev_count}")

        version_parts.append(f"+g{commit_hash}")

        if dirty_suffix:
            version_parts.append(dirty_suffix)

        result = "".join(version_parts)

        # Cache the result
        _version_cache["version"] = result
        _version_cache["base_version"] = base_version

        return result

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Git is not available or not a git repository
        result = f"{base_version}+unknown"

        # Cache the result
        _version_cache["version"] = result
        _version_cache["base_version"] = base_version

        return result
