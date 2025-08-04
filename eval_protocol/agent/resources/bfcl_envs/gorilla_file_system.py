"""Implementation of GorillaFileSystem."""

from typing import Dict, Optional, Union


class File:
    """A file in the Gorilla File System."""

    def __init__(
        self, name: str = "", content: str = "", parent: Optional["Directory"] = None
    ):  # 'Directory' as string literal
        self.name: str = name
        self.content: str = content
        self.parent: Optional["Directory"] = parent

    def __repr__(self):
        return f"<File: {self.name}, Content: '{self.content[:20]}{'...' if len(self.content) > 20 else ''}'>"

    def __eq__(self, other):
        if not isinstance(other, File):
            return False
        return self.name == other.name and self.content == other.content


class Directory:
    """A directory in the Gorilla File System."""

    def __init__(
        self,
        name: str = "",
        parent: Optional["Directory"] = None,  # Changed to string literal
        contents: Optional[Dict[str, Union[File, "Directory"]]] = None,
    ):
        self.name: str = name
        self.parent: Optional["Directory"] = parent  # Changed to string literal
        self.contents: Dict[str, Union[File, Directory]] = contents or {}

    def __repr__(self):
        parent_name = self.parent.name if self.parent else None
        return f"<Directory: {self.name}, Parent: {parent_name}, Keys: {list(self.contents.keys())}>"

    def __eq__(self, other):
        if not isinstance(other, Directory):
            return False
        return self.name == other.name and self.contents == other.contents


class GorillaFileSystem:
    """A file system for BFCL evaluation."""

    def __init__(self):
        self.root: Directory = Directory(name="workspace", parent=None)
        self.current_dir: Directory = self.root
        self.long_context: bool = False

    def _load_scenario(self, config: Dict):
        """Load the file system from configuration."""
        # self.root and self.current_dir are already initialized.
        # We will only overwrite them if loading is successful.
        if "root" in config:
            try:
                loaded_dir: Optional[Directory] = None
                root_config = config["root"]
                if isinstance(root_config, dict) and "type" in root_config:
                    loaded_dir = self._load_directory_from_config("workspace", None, root_config)
                elif isinstance(root_config, dict):  # Assuming if not 'type', it's the other YAML format
                    loaded_dir = self._load_directory_from_yaml_config("workspace", None, root_config)

                if loaded_dir:  # Check if loading returned a Directory
                    self.root = loaded_dir
                    self.current_dir = self.root
                # If loaded_dir is None, self.root and self.current_dir retain their initial default values.
            except Exception as e:
                print(f"Error loading GorillaFileSystem scenario: {e}")
                # If an exception occurred during loading, reset to a fresh default.
                self.root = Directory(name="workspace", parent=None)
                self.current_dir = self.root

        if "long_context" in config:
            self.long_context = config.get("long_context", False)

    def _load_directory_from_config(self, name: str, parent: Optional[Directory], config: Dict) -> Optional[Directory]:
        """Create a directory structure from configuration."""
        if config.get("type") == "directory":
            directory = Directory(name=name, parent=parent)
            contents: Dict[str, Union[File, Directory]] = {}
            for item_name, item_config in config.get("contents", {}).items():
                item_type = item_config.get("type")
                if item_type == "directory":
                    loaded_item = self._load_directory_from_config(item_name, directory, item_config)
                    if loaded_item:
                        contents[item_name] = loaded_item
                elif item_type == "file":
                    contents[item_name] = File(
                        name=item_name,
                        content=item_config.get("content", ""),
                        parent=directory,
                    )
            directory.contents = contents
            return directory
        return None

    def _load_directory_from_yaml_config(self, name: str, parent: Optional[Directory], config: Dict) -> Directory:
        """Create a directory structure from YAML configuration format."""
        directory = Directory(name=name, parent=parent)
        contents: Dict[str, Union[File, Directory]] = {}

        # Ensure config.get("contents") is treated as a dictionary
        config_contents = config.get("contents", {})
        if not isinstance(config_contents, dict):
            config_contents = {}  # Default to empty dict if not a dict

        for item_name, item_config in config_contents.items():
            if isinstance(item_config, dict):
                if "contents" in item_config:  # Heuristic for directory
                    loaded_subdir = self._load_directory_from_yaml_config(item_name, directory, item_config)
                    if loaded_subdir:  # Ensure it's not None, though current impl always returns Directory
                        contents[item_name] = loaded_subdir
                elif "content" in item_config:  # Heuristic for file
                    contents[item_name] = File(
                        name=item_name,
                        content=item_config.get("content", ""),
                        parent=directory,
                    )
                elif item_config.get("type") == "directory":
                    loaded_subdir = self._load_directory_from_yaml_config(item_name, directory, item_config)
                    if loaded_subdir:
                        contents[item_name] = loaded_subdir
                elif item_config.get("type") == "file":
                    contents[item_name] = File(
                        name=item_name,
                        content=item_config.get("content", ""),
                        parent=directory,
                    )
        directory.contents = contents
        return directory

    def ls(self, path: Optional[str] = None) -> Dict:
        """List directory contents."""
        target_dir: Directory = self.current_dir
        if path:
            found_node = self._find_path(path)
            if not isinstance(found_node, Directory):
                return {"error": f"Path not found or not a directory: {path}"}
            target_dir = found_node

        items: Dict[str, Dict[str, str]] = {}
        # target_dir is now guaranteed to be a Directory.
        for name, item in target_dir.contents.items():
            if isinstance(item, Directory):
                items[name] = {"type": "directory"}
            elif isinstance(item, File):
                items[name] = {"type": "file"}

        return {"current_directory": target_dir.name, "contents": items}

    def cd(self, folder: str) -> Dict:
        """Change current directory."""
        if folder == "..":
            parent_dir = self.current_dir.parent
            if parent_dir is not None:
                self.current_dir = parent_dir
                return {
                    "status": "success",
                    "message": f"Changed to {self.current_dir.name}",
                }
            else:  # Parent is None, so we are at root
                return {"status": "error", "message": "Already at root directory"}

        # self.current_dir is always a Directory. Accessing .contents is safe.
        target_item = self.current_dir.contents.get(folder)
        if isinstance(target_item, Directory):
            self.current_dir = target_item
            return {"status": "success", "message": f"Changed to {folder}"}

        return {"status": "error", "message": f"Directory {folder} not found"}

    def mkdir(self, dir_name: str) -> Dict:
        """Create a new directory."""
        # self.current_dir is always a Directory. Accessing .contents is safe.
        if dir_name in self.current_dir.contents:
            return {
                "status": "error",
                "message": f"Directory {dir_name} already exists",
            }

        self.current_dir.contents[dir_name] = Directory(name=dir_name, parent=self.current_dir)
        return {"status": "success", "message": f"Created directory {dir_name}"}

    def cat(self, file_name: str) -> Dict:
        """Display file contents."""
        # self.current_dir is always a Directory. Accessing .contents is safe.
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"status": "error", "message": f"File {file_name} not found"}

        return {
            "status": "success",
            "content": item.content,  # item is File, .content is safe
        }

    def mv(self, source: str, destination: str) -> Dict:
        """Move a file or directory."""
        source_item = self.current_dir.contents.get(source)
        if source_item is None:
            return {"status": "error", "message": f"Source {source} not found"}

        parts = destination.split("/")
        dest_name = parts[-1]
        target_dir_path = "/".join(parts[:-1])

        final_target_dir: Directory = self.current_dir
        if target_dir_path:  # If destination includes a path
            found_dir = self._find_path(target_dir_path)
            if not isinstance(found_dir, Directory):
                return {
                    "status": "error",
                    "message": f"Target directory path {target_dir_path} not found or not a directory",
                }
            final_target_dir = found_dir

        if dest_name in final_target_dir.contents:
            return {
                "status": "error",
                "message": f"Destination {destination} already exists",
            }

        # Move item
        del self.current_dir.contents[source]  # Remove from old location
        source_item.name = dest_name
        source_item.parent = final_target_dir
        final_target_dir.contents[dest_name] = source_item

        return {"status": "success", "message": f"Moved {source} to {destination}"}

    def grep(self, file_name: str, pattern: str) -> Dict:
        """Search for a pattern in a file."""
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"status": "error", "message": f"File {file_name} not found"}

        content = item.content  # item is File, .content is safe
        lines = content.split("\n")
        matches = [line for line in lines if pattern in line]

        return {"status": "success", "matches": matches, "count": len(matches)}

    def sort(self, file_name: str) -> Dict:
        """Sort the lines in a file."""
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"status": "error", "message": f"File {file_name} not found"}

        content = item.content  # item is File, .content is safe
        lines = content.split("\n")
        sorted_lines = sorted(lines)

        item.content = "\n".join(sorted_lines)  # item is File, assigning .content is safe

        return {"status": "success", "message": f"Sorted {file_name}"}

    def diff(self, file_name1: str, file_name2: str) -> Dict:
        """Compare two files."""
        item1 = self.current_dir.contents.get(file_name1)
        item2 = self.current_dir.contents.get(file_name2)

        if not isinstance(item1, File):
            return {"status": "error", "message": f"File {file_name1} not found"}
        if not isinstance(item2, File):
            return {"status": "error", "message": f"File {file_name2} not found"}

        content1 = item1.content  # item1 is File
        content2 = item2.content  # item2 is File

        if content1 == content2:
            return {
                "status": "success",
                "message": "Files are identical",
                "differences": [],
            }
        else:
            lines1 = content1.split("\n")
            lines2 = content2.split("\n")
            differences = []
            for i in range(max(len(lines1), len(lines2))):
                line1_val = lines1[i] if i < len(lines1) else None
                line2_val = lines2[i] if i < len(lines2) else None
                if line1_val != line2_val:
                    differences.append({"line": i + 1, "file1": line1_val, "file2": line2_val})
            return {
                "status": "success",
                "message": f"Found {len(differences)} differences",
                "differences": differences,
            }

    def _find_path(self, path: str) -> Optional[Union[File, Directory]]:
        """Helper to find a File or Directory by path. Returns None if not found."""
        current_node: Optional[Directory]
        parts: list[str]

        if path.startswith("/"):
            current_node = self.root
            path_str = path.strip("/")
            parts = path_str.split("/") if path_str else []
        else:
            current_node = self.current_dir
            parts = path.split("/")

        if (
            not path or path == "." or (path == "/" and not parts)
        ):  # Handle current dir or root for empty/special paths
            return self.current_dir if (not path.startswith("/")) and (path == "." or not path) else self.root

        for i, part_name in enumerate(parts):
            if current_node is None:  # Should not happen if logic is correct and current_node starts as Directory
                return None

            if not part_name:  # Skip empty parts resulting from multiple slashes e.g. /dir1//file
                if i == 0 and path.startswith("/"):  # special case for absolute path like "//file"
                    continue
                elif i > 0:
                    continue

            if part_name == "..":
                current_node = current_node.parent  # Parent can be None
                if current_node is None:  # Moved up from root
                    return None
                continue  # Successfully moved to parent

            # current_node is a Directory here.
            found_item = current_node.contents.get(part_name)

            if i == len(parts) - 1:  # This is the last part of the path
                return found_item  # Return File, Directory, or None if not found

            if isinstance(found_item, Directory):
                current_node = found_item  # Navigate into subdirectory
            else:  # Path part is not a directory or not found, and it's not the last part
                return None

        # This return is for cases like path="dir" and it's a directory, or path="/"
        return current_node
