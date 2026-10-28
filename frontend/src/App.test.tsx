import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("keeps model selection in the top bar and leaves the composer uncluttered", () => {
    const html = renderToStaticMarkup(<App />);

    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain("Model selection");
    expect(html).not.toContain('class="new-chat-button"');
    expect(html.indexOf('<header class="topbar">')).toBeLessThan(
      html.indexOf('class="hardware-fit-trigger"'),
    );
    expect(html.indexOf('class="hardware-fit-trigger"')).toBeLessThan(
      html.indexOf('class="connection-status'),
    );
    expect(html.indexOf('class="hardware-fit-trigger"')).toBeLessThan(
      html.indexOf('class="composer-row"'),
    );
    expect(html).not.toContain("Enter to send · Shift + Enter for a new line");
    expect(html).not.toContain('class="sidebar"');
    expect(html).not.toContain('class="model-control"');
    expect(html).not.toContain("Hardware fit");
  });
});
