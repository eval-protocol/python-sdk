/**
 * Minimal environment configuration for the TypeScript server
 */

/**
 * Get environment variable with fallback
 */
function getEnv(name: string, defaultValue?: string): string | undefined {
  const value = process.env[name];
  if (value && value.trim()) {
    return value.trim();
  }
  return defaultValue;
}

/**
 * Get API key with fallback chain: request -> environment
 */
export function resolveApiKey(requestApiKey?: string | null): string | null {
  if (requestApiKey && requestApiKey.trim()) {
    return requestApiKey.trim();
  }

  const envApiKey = getEnv('FIREWORKS_API_KEY');
  if (envApiKey) {
    return envApiKey;
  }

  return null;
}
