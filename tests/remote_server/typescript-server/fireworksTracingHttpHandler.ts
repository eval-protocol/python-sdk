// Minimal global declarations to avoid requiring Node type packages in this sandbox
declare const process: { env: Record<string, string | undefined> };
declare const fetch: (url: string, init?: any) => Promise<{ status: number | undefined }>;
declare const console: { log: (...args: any[]) => void };
interface MinimalMetadata {
  invocation_id: string;
  experiment_id: string;
  rollout_id: string;
  run_id: string;
  row_id: string;
}

export interface InitRequest {
  metadata: MinimalMetadata;
}

export interface FireworksStatus {
  code: number;
  message: string;
  details?: any[];
}

export interface EmitOptions {
  rolloutId?: string;
  initRequest?: InitRequest;
  message?: string;
  status?: FireworksStatus;
  tags?: string[];
  program?: string;
  extras?: Record<string, any>;
}

export class FireworksTracingHttpHandler {
  private gatewayBaseUrl: string;
  private rolloutIdEnv: string;

  constructor(gatewayBaseUrl?: string, rolloutIdEnv: string = "EP_ROLLOUT_ID") {
    const base =
      gatewayBaseUrl ||
      process.env["FW_TRACING_GATEWAY_BASE_URL"] ||
      process.env["GATEWAY_URL"] ||
      "https://tracing.fireworks.ai";
    this.gatewayBaseUrl = String(base).replace(/\/$/, "");
    this.rolloutIdEnv = rolloutIdEnv;
  }

  async emit(opts: EmitOptions): Promise<void> {
    try {
      const rolloutId = this.resolveRolloutId(opts);
      if (!rolloutId) return;

      const payload = this.buildPayload(rolloutId, opts);
      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      const apiKey = process.env["FIREWORKS_API_KEY"];
      if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

      const url = `${this.gatewayBaseUrl}/logs`;
      const respStatus = await this.postJson(url, payload, headers);
      if (respStatus === 404) {
        const altUrl = `${this.gatewayBaseUrl}/v1/logs`;
        await this.postJson(altUrl, payload, headers);
      }
    } catch (_err) {
      // Never throw from logging
    }
  }

  private resolveRolloutId(opts: EmitOptions): string | undefined {
    if (opts.rolloutId) return opts.rolloutId;
    if (opts.initRequest?.metadata?.rollout_id) return opts.initRequest.metadata.rollout_id;
    const envVal = process.env[this.rolloutIdEnv];
    return envVal && envVal.trim() ? envVal : undefined;
  }

  private buildPayload(rolloutId: string, opts: EmitOptions): Record<string, any> {
    const message = opts.message ?? "";
    const tags = new Set<string>(opts.tags ?? []);
    // Always include rollout tag
    tags.add(`rollout_id:${rolloutId}`);

    // If initRequest provided, include standard tags
    if (opts.initRequest) {
      const md = opts.initRequest.metadata;
      tags.add(`invocation_id:${md.invocation_id}`);
      tags.add(`experiment_id:${md.experiment_id}`);
      tags.add(`rollout_id:${md.rollout_id}`);
      tags.add(`run_id:${md.run_id}`);
      tags.add(`row_id:${md.row_id}`);
    }

    const timestamp = new Date().toISOString();
    const program = opts.program ?? "eval_protocol";

    return {
      program,
      status: opts.status,
      message,
      tags: Array.from(tags),
      extras: {
        timestamp,
      },
    };
  }

  private async postJson(url: string, body: any, headers: Record<string, string>): Promise<number | undefined> {
    try {
      const fetchFn: any = (globalThis as any).fetch;
      if (!fetchFn) return undefined;
      const debug = process.env["EP_DEBUG"] === "true";
      if (debug) {
        try {
          const msg = typeof body?.message === "string" ? body.message.slice(0, 80) : String(body?.message ?? "");
          const tagCount = Array.isArray(body?.tags) ? body.tags.length : 0;
          // eslint-disable-next-line no-console
          console.log(`[FW_LOG] POST ${url} tags=${tagCount} msg=${msg}`);
        } catch {
          // ignore
        }
      }
      const resp = await fetchFn(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      if (debug) {
        try {
          // eslint-disable-next-line no-console
          console.log(`[FW_LOG] resp=${resp?.status}`);
        } catch {
          // ignore
        }
      }
      return resp?.status;
    } catch {
      return undefined;
    }
  }
}
