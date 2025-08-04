import argparse
import secrets


def generate_api_key(length: int = 32) -> str:
    """
    Generates a cryptographically strong random string to be used as an API key.

    Args:
        length: The desired length of the API key in bytes.
                The resulting hex string will be twice this length.
                Default is 32 bytes, resulting in a 64-character hex string.

    Returns:
        A hex-encoded secure random string.
    """
    return secrets.token_hex(length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a secure API key.")
    parser.add_argument(
        "--length",
        type=int,
        default=32,
        help="Length of the key in bytes (default: 32, produces a 64-char hex string).",
    )
    args = parser.parse_args()

    api_key = generate_api_key(args.length)
    print(api_key)
