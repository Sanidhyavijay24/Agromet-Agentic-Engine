/**
 * @file App.test.ts
 * @description Frontend smoke test
 * @module frontend/tests
 */

import { describe, expect, it } from "bun:test";
import React from "react";
import { renderToString } from "react-dom/server";
import { App } from "./App";

describe("Frontend Clean Slate Smoke Test", () => {
  it("renders the root application shell without error", () => {
    const html = renderToString(React.createElement(App));
    expect(html).toContain("Agromet Agentic Engine");
  });
});
