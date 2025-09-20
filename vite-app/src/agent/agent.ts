import OpenAI from "openai";

export class Agent {
  client: OpenAI;
  /**
   * This is an agent that has access to tools that will allow it to search
   * through eval results
   */
  constructor() {
    this.client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });
  }
}
