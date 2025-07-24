"""Mock implementation of GorillaFileSystem."""

from typing import Dict, Optional, Union


class File:
    """A mock file in the Gorilla File System."""

    def __init__(self, name: str = "", content: str = "", parent: Optional["Directory"] = None):
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
    """A mock directory in the Gorilla File System."""

    def __init__(
        self,
        name: str = "",
        parent: Optional["Directory"] = None,  # Changed to string literal
        contents: Optional[Dict[str, Union[File, "Directory"]]] = None,
    ):
        self.name: str = name
        self.parent: Optional["Directory"] = parent  # Changed to string literal
        self.contents: Dict[str, Union[File, "Directory"]] = contents or {}

    def __repr__(self):
        parent_name = self.parent.name if self.parent and self.parent.name else None
        return f"<Directory: {self.name}, Parent: {parent_name}, Keys: {list(self.contents.keys())}>"

    def __eq__(self, other):
        if not isinstance(other, Directory):
            return False
        return self.name == other.name and self.contents == other.contents


class GorillaFileSystem:
    """A mock file system for BFCL tests."""

    def __init__(self):
        self.root: Directory = Directory(name="workspace", parent=None)  # Initialize with a Directory
        self.current_dir: Directory = self.root  # Initialize with a Directory
        self.long_context: bool = False

    def _load_scenario(self, config: Dict):
        """Load the file system from configuration."""
        if "root" in config:
            try:
                loaded_dir: Optional[Directory] = None
                root_config = config["root"]
                if isinstance(root_config, dict) and "type" in root_config:
                    loaded_dir = self._load_directory_from_config("workspace", None, root_config)
                elif isinstance(root_config, dict):
                    loaded_dir = self._load_directory_from_yaml_config("workspace", None, root_config)

                if loaded_dir:
                    self.root = loaded_dir
                    self.current_dir = self.root
                # If loading fails or loaded_dir is None, keep the default initialized root/current_dir
            except Exception as e:
                print(f"Error loading GorillaFileSystem scenario: {e}")
                # Fallback to a fresh default if loading caused an exception
                self.root = Directory(name="workspace", parent=None)
                self.current_dir = self.root

        if "long_context" in config:
            self.long_context = config.get("long_context", False)

    def _load_directory_from_config(
        self, name: str, parent: Optional["Directory"], config: Dict
    ) -> Optional[Directory]:
        """Create a directory structure from configuration."""
        if config.get("type") == "directory":
            directory = Directory(name=name, parent=parent)
            contents: Dict[str, Union[File, Directory]] = {}
            for item_name, item_config in config.get("contents", {}).items():
                item_type = item_config.get("type")
                if item_type == "directory":
                    loaded_item = self._load_directory_from_config(item_name, directory, item_config)
                    if loaded_item:  # Check if it's not None
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

    def _load_directory_from_yaml_config(self, name: str, parent: Optional["Directory"], config: Dict) -> Directory:
        """Create a directory structure from YAML configuration format."""
        directory = Directory(name=name, parent=parent)
        contents: Dict[str, Union[File, "Directory"]] = {}

        config_contents = config.get("contents", {})
        if not isinstance(config_contents, dict):
            config_contents = {}

        for item_name, item_config in config_contents.items():
            if isinstance(item_config, dict):
                if "contents" in item_config:
                    loaded_subdir = self._load_directory_from_yaml_config(item_name, directory, item_config)
                    if loaded_subdir:
                        contents[item_name] = loaded_subdir
                elif "content" in item_config:
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
            if not isinstance(found_node, Directory):  # Check if it's a Directory instance
                return {"error": f"Path not found or not a directory: {path}"}
            target_dir = found_node

        # target_dir is now guaranteed to be a Directory
        items: Dict[str, Dict[str, str]] = {}
        for (
            name_key,
            item_val,
        ) in target_dir.contents.items():  # Use different var names
            if isinstance(item_val, Directory):
                items[name_key] = {"type": "directory"}
            elif isinstance(item_val, File):
                items[name_key] = {"type": "file"}

        return {"current_directory": target_dir.name, "contents": items}

    def cd(self, folder: str) -> Dict:
        """Change current directory."""
        # self.current_dir is guaranteed to be a Directory
        if folder == "..":
            parent_dir = self.current_dir.parent
            if parent_dir is not None:  # Explicit check for None
                self.current_dir = parent_dir
                return {
                    "status": "success",
                    "message": f"Changed to {self.current_dir.name}",  # Safe: self.current_dir is Directory
                }
            return {"status": "error", "message": "Already at root directory"}

        target_item = self.current_dir.contents.get(folder)
        if isinstance(target_item, Directory):  # Check if it's a Directory instance
            self.current_dir = target_item
            return {"status": "success", "message": f"Changed to {folder}"}

        return {"status": "error", "message": f"Directory {folder} not found"}

    def mkdir(self, dir_name: str) -> Dict:
        """Create a new directory."""
        # self.current_dir is guaranteed to be a Directory
        if dir_name in self.current_dir.contents:
            return {
                "status": "error",
                "message": f"Directory {dir_name} already exists",
            }

        self.current_dir.contents[dir_name] = Directory(name=dir_name, parent=self.current_dir)
        return {"status": "success", "message": f"Created directory {dir_name}"}

    def cat(self, file_name: str) -> Dict:
        """Display file contents."""
        # self.current_dir is guaranteed to be a Directory
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):  # Check if it's a File instance
            return {"status": "error", "message": f"File {file_name} not found"}

        # item is now guaranteed to be a File
        return {
            "status": "success",
            "content": item.content,
        }

    def mv(self, source: str, destination: str) -> Dict:
        """Move a file or directory."""
        # self.current_dir is guaranteed to be a Directory
        source_item = self.current_dir.contents.get(source)
        if source_item is None:
            return {"status": "error", "message": f"Source {source} not found"}

        parts = destination.split("/")
        dest_name = parts[-1]
        target_dir_path = "/".join(parts[:-1])

        final_target_dir: Directory = self.current_dir
        if target_dir_path:
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

        del self.current_dir.contents[source]
        source_item.name = dest_name  # source_item is File or Directory, .name is safe
        source_item.parent = final_target_dir  # .parent is safe
        final_target_dir.contents[dest_name] = source_item

        return {"status": "success", "message": f"Moved {source} to {destination}"}

    def grep(self, file_name: str, pattern: str) -> Dict:
        """Search for a pattern in a file."""
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"status": "error", "message": f"File {file_name} not found"}

        # item is File
        content = item.content
        lines = content.split("\n")
        matches = [line for line in lines if pattern in line]

        return {"status": "success", "matches": matches, "count": len(matches)}

    def sort(self, file_name: str) -> Dict:
        """Sort the lines in a file."""
        item = self.current_dir.contents.get(file_name)
        if not isinstance(item, File):
            return {"status": "error", "message": f"File {file_name} not found"}

        # item is File
        content = item.content
        lines = content.split("\n")
        sorted_lines = sorted(lines)
        item.content = "\n".join(sorted_lines)

        return {"status": "success", "message": f"Sorted {file_name}"}

    def diff(self, file_name1: str, file_name2: str) -> Dict:
        """Compare two files."""
        item1 = self.current_dir.contents.get(file_name1)
        item2 = self.current_dir.contents.get(file_name2)

        if not isinstance(item1, File):
            return {"status": "error", "message": f"File {file_name1} not found"}
        if not isinstance(item2, File):
            return {"status": "error", "message": f"File {file_name2} not found"}

        # item1 and item2 are Files
        content1 = item1.content
        content2 = item2.content

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
            current_node = self.root  # self.root is Directory
            path_str = path.strip("/")
            parts = path_str.split("/") if path_str else []
        else:
            current_node = self.current_dir  # self.current_dir is Directory
            parts = path.split("/")

        if not path or path == "." or (path == "/" and not parts):
            return self.current_dir if (not path.startswith("/")) and (path == "." or not path) else self.root

        for i, part_name in enumerate(parts):
            if current_node is None:  # This check is important if parent can lead to None
                return None

            if not part_name:
                if i == 0 and path.startswith("/"):
                    continue
                elif i > 0:
                    continue

            if part_name == "..":
                current_node = current_node.parent  # current_node.parent can be None
                if current_node is None:
                    return None
                continue

            # current_node is confirmed to be a Directory here (or loop would have exited)
            found_item = current_node.contents.get(part_name)

            if i == len(parts) - 1:
                return found_item

            if isinstance(found_item, Directory):
                current_node = found_item
            else:
                return None

        return current_node
