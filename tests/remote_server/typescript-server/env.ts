import * as dotenv from "dotenv";

// Helper to resolve the root of the repo (for .env loading, etc.)
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

// Returns the absolute path to the root of the repo (where .git or .env is found)
function getRepoRoot(): string {
  // __dirname is not available in ES modules, so use fileURLToPath
  const currentDir = path.dirname(fileURLToPath(import.meta.url));
  let dir = currentDir;
  while (true) {
    if (
      fs.existsSync(path.join(dir, ".git")) ||
      fs.existsSync(path.join(dir, ".env"))
    ) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  // Fallback to current directory if not found
  return currentDir;
}

export const REPO_ROOT = getRepoRoot();

// Load environment variables from .env at the root of the repo
dotenv.config({ path: path.join(REPO_ROOT, ".env") });
