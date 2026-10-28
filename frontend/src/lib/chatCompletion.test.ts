import { describe, expect, it } from "vitest";
import { readChatCompletionStream } from "./chatCompletion";

describe("chat completion SSE reader", () => {
  it("parses events split across chunks and preserves split UTF-8 characters", async () => {
    const bytes = new TextEncoder().encode(
      'data: {"choices":[{"delta":{"content":"café"}}]}\n\ndata: [DONE]\n\n'
    );
    const split = bytes.indexOf(0xc3) + 1;
    const response = new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(bytes.slice(0, split));
          controller.enqueue(bytes.slice(split, split + 7));
          controller.enqueue(bytes.slice(split + 7));
          controller.close();
        }
      }),
      { headers: { "Content-Type": "text/event-stream" } }
    );
    const chunks: string[] = [];

    await readChatCompletionStream(response, (content) => chunks.push(content));

    expect(chunks.join("")).toBe("café");
  });
});
