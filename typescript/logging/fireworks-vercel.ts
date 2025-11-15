/**
 * Vercel-specific helper to wire Fireworks logging into serverless handlers.
 *
 * Usage:
 *   export default withFireworksLogging(async (req, res) => { ... })
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { waitUntil } from '@vercel/functions';
import { setWaitUntil } from './logger.js';

export type VercelHandler<T = any> = (req: VercelRequest, res: VercelResponse) => T | Promise<T>;

export function withFireworksLogging<T = any>(handler: VercelHandler<T>): VercelHandler<T> {
  return async (req, res) => {
    // Hook up Vercel waitUntil for this invocation so logging
    // can flush to Fireworks even after the HTTP response is sent.
    setWaitUntil(waitUntil);
    return handler(req, res);
  };
}
