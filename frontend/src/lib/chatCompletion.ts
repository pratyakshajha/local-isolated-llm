export async function readChatCompletionStream(
  response: Response,
  onContent: (content: string) => void
): Promise<void> {
  if (!response.body) throw new Error("The server returned an empty response.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const readLine = (line: string): boolean => {
    line = line.endsWith("\r") ? line.slice(0, -1) : line;
    if (!line.startsWith("data:")) return false;
    const data = line.slice(5).trimStart();
    if (data === "[DONE]") return true;
    const event = JSON.parse(data) as {
      choices?: Array<{ delta?: { content?: unknown } }>;
      error?: { message?: string };
    };
    if (event.error?.message) throw new Error(event.error.message);
    const content = event.choices?.[0]?.delta?.content;
    if (typeof content === "string") onContent(content);
    return false;
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let newline = buffer.indexOf("\n");
      while (newline >= 0) {
        if (readLine(buffer.slice(0, newline))) return;
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf("\n");
      }
      if (done) {
        if (buffer) readLine(buffer);
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
