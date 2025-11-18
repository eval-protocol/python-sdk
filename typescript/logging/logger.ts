/**
 * Winston logger configuration with Fireworks tracing transport.
 */

import winston from 'winston';
import { FireworksTransport } from './fireworks-transport.js';

// Global reference to waitUntil function
let globalWaitUntil: ((promise: Promise<any>) => void) | undefined;

// Set waitUntil function (called from Vercel handler or host)
export function setWaitUntil(waitUntil: (promise: Promise<any>) => void) {
  globalWaitUntil = waitUntil;
}

export const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new FireworksTransport({
      waitUntil: (promise: Promise<any>) => globalWaitUntil?.(promise)
    })
  ]
});

/**
 * Create a child logger with rollout_id context.
 */
export function createRolloutLogger(rolloutId: string, name: string = 'init'): winston.Logger {
  return logger.child({
    rollout_id: rolloutId,
    logger_name: `${name}.${rolloutId}`
  });
}
