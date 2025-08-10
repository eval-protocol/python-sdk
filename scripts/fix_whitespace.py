import os

TEXT_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".json",
    ".rst",
}


def is_text_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in TEXT_EXTS


def fix_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return False

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return False

    lines = text.splitlines()
    changed = False

    # Strip trailing whitespace on each line
    new_lines = []
    for line in lines:
        new_line = line.rstrip()
        if new_line != line:
            changed = True
        new_lines.append(new_line)

    # Ensure newline at EOF
    new_text = "\n".join(new_lines) + "\n"
    if not text.endswith("\n"):
        changed = True

    if changed:
        with open(path, "wb") as f:
            f.write(new_text.encode("utf-8"))
    return changed


def main():
    root = os.getcwd()
    total = 0
    changed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip virtualenvs and build artifacts
        parts = dirpath.split(os.sep)
        if any(p in {".git", ".venv", "build", "dist", "node_modules"} for p in parts):
            continue
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if is_text_file(path):
                total += 1
                if fix_file(path):
                    changed += 1
    print(f"Scanned {total} files; normalized whitespace in {changed} files")


if __name__ == "__main__":
    main()

