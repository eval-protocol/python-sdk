/**
 * Winston transport that sends logs to Fireworks tracing gateway.
 */

import Transport from 'winston-transport';
import type { TransformableInfo } from 'logform';
const LEVEL = Symbol.for('level');

interface FireworksLogInfo extends TransformableInfo {
  rollout_id?: string;
  experiment_id?: string;
  run_id?: string;
  rollout_ids?: string[];
  status?: any;
  program?: string;
  logger_name?: string;
  extras?: Record<string, any>;
  [key: string]: any;
}

interface StatusInfo {
  code?: number;
  message?: string;
  details?: any[];
}

interface FireworksPayload {
  program: string;
  status?: StatusInfo | null;
  message: string;
  tags: string[];
  extras: {
    logger_name: string;
    level: string;
    timestamp: string;
    [key: string]: any;
  };
}

export class FireworksTransport extends Transport {
  private gatewayBaseUrl: string;
  private rolloutIdEnv: string;
  private apiKey?: string;
  private waitUntil?: (promise: Promise<any>) => void;

  constructor(opts: {
    gatewayBaseUrl?: string;
    rolloutIdEnv?: string;
    waitUntil?: (promise: Promise<any>) => void;
  } = {}) {
    super();

    this.gatewayBaseUrl =
      opts.gatewayBaseUrl ||
      process.env.FW_TRACING_GATEWAY_BASE_URL ||
      'https://tracing.fireworks.ai';

    this.rolloutIdEnv = opts.rolloutIdEnv || 'EP_ROLLOUT_ID';
    this.apiKey = process.env.FIREWORKS_API_KEY;
    this.waitUntil = opts.waitUntil;
  }

  log(info: FireworksLogInfo, callback: () => void) {
    setImmediate(() => {
      this.emit('logged', info);
    });

    const sendPromise = this.sendToFireworks(info).catch((error) => {
      this.emit('error', error);
    });

    // Use waitUntil for ALL logs when available so Fireworks logging
    // can complete even after the HTTP response is sent.
    if (this.waitUntil) {
      this.waitUntil(sendPromise);
    }

    callback();
  }

  private async sendToFireworks(info: FireworksLogInfo): Promise<void> {
    if (!this.gatewayBaseUrl) {
      return;
    }

    const rolloutId = this.getRolloutId(info);
    if (!rolloutId) {
      return;
    }

    const payload = this.buildPayload(info, rolloutId);
    const baseUrl = this.gatewayBaseUrl.replace(/\/$/, '');
    const url = `${baseUrl}/logs`;

    let payloadJson: string;
    try {
      payloadJson = JSON.stringify(payload);
    } catch (e: any) {
      const msg = `Fireworks logging payload is not JSON-serializable (rollout_id=${rolloutId}): ${e?.message || e}`;
      console.error(`[FW_LOG] ${msg}`);
      this.emit('error', new Error(msg));
      return;
    }

    // Debug logging
    if (process.env.EP_DEBUG === 'true') {
      const tagsLen = Array.isArray(payload.tags) ? payload.tags.length : 0;
      const msgPreview = typeof payload.message === 'string'
        ? payload.message.substring(0, 80)
        : payload.message;
      const payloadSize = payloadJson.length;
      const hasStatus = !!payload.status;
      console.log(`[FW_LOG] POST ${url} rollout_id=${rolloutId} tags=${tagsLen} msg=${msgPreview} size=${payloadSize} hasStatus=${hasStatus}`);
    }

    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'User-Agent': 'winston-fireworks-transport/1.0.0',
      };

      if (this.apiKey) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: payloadJson,
        // No timeout signal for compatibility
      });

      if (process.env.EP_DEBUG === 'true') {
        console.log(`[FW_LOG] resp=${response.status} for rollout_id=${rolloutId}`);
      }

      // Fallback to /v1/logs if /logs is not found
      if (response.status === 404) {
        const altUrl = `${baseUrl}/v1/logs`;

        if (process.env.EP_DEBUG === 'true') {
          const tagsLen = Array.isArray(payload.tags) ? payload.tags.length : 0;
          console.log(`[FW_LOG] RETRY POST ${altUrl} rollout_id=${rolloutId} tags=${tagsLen}`);
        }

        const retryResponse = await fetch(altUrl, {
          method: 'POST',
          headers,
          body: payloadJson,
          // No timeout signal for compatibility
        });

        if (process.env.EP_DEBUG === 'true') {
          console.log(`[FW_LOG] retry resp=${retryResponse.status}`);
        }
      }

    } catch (error: any) {
      // Silently handle errors - logging should not break the application
      if (process.env.EP_DEBUG === 'true') {
        console.error(`[FW_LOG] Error sending to Fireworks:`, error.message);
        console.error(`[FW_LOG] Payload was:`, payloadJson);
      }
    }
  }

  private getRolloutId(info: FireworksLogInfo): string | null {
    // Check if rollout_id is in the log info
    if (info.rollout_id && typeof info.rollout_id === 'string') {
      return info.rollout_id;
    }

    // Fallback to environment variable
    return process.env[this.rolloutIdEnv] || null;
  }

  private getStatusInfo(info: FireworksLogInfo): StatusInfo | null {
    if (!info.status) {
      return null;
    }

    const status = info.status;

    // Handle Status class instances (with code and message properties)
    if (typeof status === 'object' && status !== null && 'code' in status && 'message' in status) {
      return {
        code: typeof status.code === 'number' ? status.code : undefined,
        message: typeof status.message === 'string' ? status.message : undefined,
        details: Array.isArray(status.details) ? status.details : [],
      };
    }

    return null;
  }

  private buildPayload(info: FireworksLogInfo, rolloutId: string): FireworksPayload {
    const timestamp = new Date().toISOString();
    // Ensure message is always a string for Fireworks payload
    const message: string = typeof info.message === 'string' ? info.message : '';
    const level = (info as any)[LEVEL] || info.level || 'info';

    const tags: string[] = [`rollout_id:${rolloutId}`];

    // Optional additional tags
    if (info.experiment_id && typeof info.experiment_id === 'string') {
      tags.push(`experiment_id:${info.experiment_id}`);
    }
    if (info.run_id && typeof info.run_id === 'string') {
      tags.push(`run_id:${info.run_id}`);
    }

    // Groupwise list of rollout_ids
    if (Array.isArray(info.rollout_ids)) {
      for (const rid of info.rollout_ids) {
        if (typeof rid === 'string') {
          tags.push(`rollout_id:${rid}`);
        }
      }
    }

    const program = (typeof info.program === 'string' ? info.program : null) || 'eval_protocol';

    const extraInput =
      info.extras && typeof info.extras === 'object' && !Array.isArray(info.extras)
        ? info.extras
        : {};

    return {
      program,
      status: this.getStatusInfo(info),
      message,
      tags,
      extras: {
        ...extraInput,
        logger_name: info.logger_name || 'winston',
        level: level.toUpperCase(),
        timestamp,
      },
    };
  }
}
