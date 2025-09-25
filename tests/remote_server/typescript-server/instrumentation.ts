import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import "./env.js";

const sdk = new NodeSDK({
  spanProcessors: [
    new LangfuseSpanProcessor({
      publicKey: process.env["LANGFUSE_PUBLIC_KEY"]!,
      secretKey: process.env["LANGFUSE_SECRET_KEY"]!,
      baseUrl: process.env["LANGFUSE_HOST"]!,
    }),
  ],
});

sdk.start();
