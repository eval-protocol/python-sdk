/**
 * Winston logger configuration with Fireworks tracing transport.
 */

import winston from 'winston';
import { FireworksTransport } from './fireworks-transport.js';

type RolloutLoggerOptions = {
  gatewayBaseUrl?: string;
  apiKey?: string;
  name?: string;
};

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
export function createRolloutLogger(
  rolloutId: string,
  nameOrOptions: string | RolloutLoggerOptions = 'init'
): winston.Logger {
  const options = typeof nameOrOptions === 'string' ? { name: nameOrOptions } : nameOrOptions;
  const name = options.name || 'init';
  const defaultMeta = {
    rollout_id: rolloutId,
    logger_name: `${name}.${rolloutId}`
  };

  if (options.gatewayBaseUrl || options.apiKey) {
    return winston.createLogger({
      level: 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
      ),
      defaultMeta,
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
          )
        }),
        new FireworksTransport({
          gatewayBaseUrl: options.gatewayBaseUrl,
          apiKey: options.apiKey,
          waitUntil: (promise: Promise<any>) => globalWaitUntil?.(promise)
        })
      ]
    });
  }

  return logger.child(defaultMeta);
}
