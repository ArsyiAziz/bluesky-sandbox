import { useEffect, useMemo, useState } from "react";
import Editor from "@monaco-editor/react";
import { api, type CompletionContext, type CompletionSymbol, type PythonMember, type SpecDict, type ValidateResult } from "../api";
import { Picker } from "./panel/Picker";

// Module-level so the Monaco completion provider (registered once) always reads
// the latest env components even after the Code tab remounts.
let latestCompletionContext: CompletionContext | null = null;
let completionsRegistered = false;
let semanticTokensDidChange: any = null;
const moduleMemberCache = new Map<string, Promise<PythonMember[]>>();

const CUSTOM_FIELDS_TEMPLATE = `"""Custom observation/action fields for this design.

Classes in this module are referenced as import strings, e.g.
\`custom_fields:MyField\`. Use the designer's field configuration modal to set
constructor bounds and normalizers for each referenced class.
"""

from __future__ import annotations

from dataclasses import dataclass

import bluesky as bs

from bluesky_sandbox.fields.base import (
    ActionField, ActionMeta, ActionMode, ControlAxis,
    ObsField, ObsMeta, ObsQuantity, Unit,
)
`;

const TASK_INFO_BODY_TEMPLATE = `task = info["task"]
task["metric"] = 0.0
`;

type TaskInfoEntry = {
  name: string;
  body: string;
};

type TaskInfoType = {
  name: string;
  doc?: string;
  category?: string;
  params?: { name: string; type?: string; required?: boolean; default?: any }[];
  scaffold?: {
    name: string;
    provider_var: string;
    setup: string;
    body: string;
  };
};

type ExpressionInfo = {
  symbol?: CompletionSymbol;
  members: CompletionSymbol[];
  nestedMembers?: Record<string, CompletionSymbol[]>;
};

function moduleMembers(moduleName: string): Promise<PythonMember[]> {
  const cached = moduleMemberCache.get(moduleName);
  if (cached) return cached;
  const promise = api
    .pythonModuleMembers(moduleName)
    .then((result) => result.members)
    .catch(() => {
      moduleMemberCache.delete(moduleName);
      return [];
    });
  moduleMemberCache.set(moduleName, promise);
  return promise;
}

function editorContext(path: string): "hook_setup" | "hook" | "task_info_setup" | "task_info" | "custom_code" | "other" {
  if (path.includes("hook_setup")) return "hook_setup";
  if (path.includes("hook_")) return "hook";
  if (path.includes("task_info_setup")) return "task_info_setup";
  if (path.includes("task_info_")) return "task_info";
  if (path.endsWith("custom_fields.py") || path.includes("custom_field_")) return "custom_code";
  return "other";
}

function setupScope(context: ReturnType<typeof editorContext>): { symbols: CompletionSymbol[]; imports: Record<string, string> } {
  const completionContext = latestCompletionContext?.ok ? latestCompletionContext : null;
  if (context === "hook" || context === "hook_setup") {
    return completionContext?.hook_setup ?? { symbols: [], imports: {} };
  }
  if (context === "task_info" || context === "task_info_setup") {
    return completionContext?.task_info_setup ?? { symbols: [], imports: {} };
  }
  return { symbols: [], imports: {} };
}

function hookNameFromPath(path: string): string | null {
  const match = path.match(/hook_([^/]+)\.py$/);
  if (!match || match[1] === "setup") return null;
  return match[1];
}

function completionKind(symbolKind: string | undefined, K: any): number {
  if (symbolKind === "function") return K.Function;
  if (symbolKind === "class") return K.Class;
  if (symbolKind === "module") return K.Module;
  if (symbolKind === "property") return K.Property;
  if (symbolKind === "field") return K.Field;
  if (symbolKind === "variable") return K.Variable;
  return K.Value;
}

function semanticTokenType(symbolKind: string | undefined): string {
  if (symbolKind === "module") return "namespace";
  if (symbolKind === "class") return "class";
  if (symbolKind === "function") return "function";
  if (symbolKind === "property" || symbolKind === "field") return "property";
  return "variable";
}

function semanticSymbolsForModel(model: any): Map<string, string> {
  const completionContext = latestCompletionContext?.ok ? latestCompletionContext : null;
  const catalog = expressionCatalog(completionContext);
  const path = String(model.uri?.path ?? model.uri?.toString?.() ?? "");
  const context = editorContext(path);
  const symbols = new Map<string, string>();
  const addSymbol = (symbol: CompletionSymbol | undefined, fallbackType?: string) => {
    if (!symbol?.name || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(symbol.name)) return;
    symbols.set(symbol.name, fallbackType ?? semanticTokenType(symbol.kind));
  };
  const scope = setupScope(context);
  scope.symbols.forEach((symbol) => addSymbol(symbol));
  Object.keys(scope.imports).forEach((name) => symbols.set(name, "namespace"));

  const hookName = context === "hook" ? hookNameFromPath(path) : null;
  const activeHookContext = hookName ? completionContext?.hooks?.[hookName] : undefined;
  const params = context === "task_info"
    ? completionContext?.task_info?.params
    : activeHookContext?.params;
  (params ?? []).forEach((symbol) => addSymbol(symbol, "parameter"));

  const activeMembers = context === "task_info"
    ? completionContext?.task_info?.members
    : activeHookContext?.members;
  Object.keys(activeMembers ?? {}).forEach((name) => {
    symbols.set(name, name === "self" ? "variable" : "parameter");
  });
  expressionAliasesForModel(model, catalog).forEach((_, name) => {
    symbols.set(name, "variable");
  });

  return symbols;
}

function activeMembersForModel(model: any, completionContext: CompletionContext | null): Record<string, CompletionSymbol[]> {
  if (!completionContext) return {};
  const path = String(model.uri?.path ?? model.uri?.toString?.() ?? "");
  const context = editorContext(path);
  if (context === "task_info") return completionContext.task_info?.members ?? {};
  if (context !== "hook") return {};
  const hookName = hookNameFromPath(path);
  return hookName ? completionContext.hooks?.[hookName]?.members ?? {} : {};
}

function symbolTokenType(symbols: CompletionSymbol[] | undefined, name: string): string | null {
  const symbol = symbolByName(symbols, name);
  return symbol ? semanticTokenType(symbol.kind) : null;
}

function expressionCatalog(completionContext: CompletionContext | null): Map<string, ExpressionInfo> {
  const catalog = new Map<string, ExpressionInfo>();
  if (!completionContext) return catalog;
  const addExpression = (expression: string, info: ExpressionInfo) => {
    catalog.set(expression, info);
    catalog.set(expression.split('"').join("'"), info);
  };
  if (completionContext.airspace_result_members) {
    addExpression("context.airspace", {
      symbol: {
        name: "context.airspace",
        kind: "property",
        detail: "RegionResult",
        doc: "Airspace query result for this aircraft.",
        insert: "context.airspace",
      },
      members: completionContext.airspace_result_members,
      nestedMembers: completionContext.airspace_result_nested_members,
    });
  }
  Object.entries(completionContext.query_result_members ?? {}).forEach(([name, members]) => {
    addExpression(`context.query("${name}")`, {
      symbol: symbolByName(completionContext.query_calls, `context.query("${name}")`),
      members,
      nestedMembers: completionContext.query_result_nested_members?.[name],
    });
  });
  Object.entries(completionContext.queryable_members ?? {}).forEach(([name, members]) => {
    addExpression(`context.queryable("${name}")`, {
      symbol: symbolByName(completionContext.queryables, `"${name}"`),
      members,
    });
  });
  return catalog;
}

function expressionAliasesFromText(text: string, catalog: Map<string, ExpressionInfo>): Map<string, string> {
  const aliases = new Map<string, string>();
  const assignment = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(?:#.*)?$/gm;
  for (const match of text.matchAll(assignment)) {
    const rhs = match[2].trim();
    if (catalog.has(rhs)) aliases.set(match[1], rhs);
  }
  return aliases;
}

function modelTextBefore(model: any, lineNumber: number, column: number): string {
  const lines: string[] = [];
  for (let line = 1; line <= lineNumber; line += 1) {
    const content = String(model.getLineContent(line));
    lines.push(line === lineNumber ? content.slice(0, column - 1) : content);
  }
  return lines.join("\n");
}

function expressionAliasesForModel(model: any, catalog: Map<string, ExpressionInfo>): Map<string, string> {
  return expressionAliasesFromText(String(model.getValue?.() ?? ""), catalog);
}

function expressionAliasesBefore(model: any, position: any, catalog: Map<string, ExpressionInfo>): Map<string, string> {
  return expressionAliasesFromText(modelTextBefore(model, position.lineNumber, position.column), catalog);
}

function expressionEndingAt(source: string): string | null {
  const match = source.match(/([A-Za-z_][A-Za-z0-9_]*(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:\(\s*["'][^"']+["']\s*\)))*)$/);
  return match?.[1] ?? null;
}

function completionExpression(linePrefix: string): string | null {
  const beforePartial = linePrefix.replace(/[A-Za-z_][A-Za-z0-9_]*$/, "");
  if (!beforePartial.endsWith(".")) return null;
  return expressionEndingAt(beforePartial.slice(0, -1).trimEnd());
}

function resolveExpressionInfo(
  expression: string,
  catalog: Map<string, ExpressionInfo>,
  aliases: Map<string, string>,
): ExpressionInfo | undefined {
  const aliasedExpression = aliases.get(expression);
  if (aliasedExpression) return catalog.get(aliasedExpression);
  return catalog.get(expression);
}

function resolveExpressionMembers(
  expression: string,
  catalog: Map<string, ExpressionInfo>,
  aliases: Map<string, string>,
): CompletionSymbol[] {
  const direct = resolveExpressionInfo(expression, catalog, aliases);
  if (direct) return attributeSymbols(direct.members);
  const dot = expression.lastIndexOf(".");
  if (dot < 0) return [];
  const base = expression.slice(0, dot);
  const member = expression.slice(dot + 1);
  const baseInfo = resolveExpressionInfo(base, catalog, aliases);
  return attributeSymbols(baseInfo?.nestedMembers?.[member]);
}

function dottedSemanticTokenType(
  line: string,
  startColumn: number,
  name: string,
  activeMembers: Record<string, CompletionSymbol[]>,
  importedMembers: Map<string, Map<string, string>>,
  catalog: Map<string, ExpressionInfo>,
  aliases: Map<string, string>,
): string | null {
  const prefix = line.slice(0, startColumn);
  if (!prefix.endsWith(".")) return null;
  const expression = expressionEndingAt(prefix.slice(0, -1).trimEnd());
  if (expression) {
    const expressionTokenType = symbolTokenType(
      resolveExpressionMembers(expression, catalog, aliases),
      name,
    );
    if (expressionTokenType) return expressionTokenType;
  }

  const objectMember = prefix.match(/([A-Za-z_][A-Za-z0-9_]*)\.$/);
  if (!objectMember) return null;
  const root = objectMember[1];
  return symbolTokenType(activeMembers[root], name) ?? importedMembers.get(root)?.get(name) ?? null;
}

function definitionSemanticTokenType(line: string, startColumn: number, name: string): string | null {
  const before = line.slice(0, startColumn);
  if (new RegExp(`^\\s*class\\s+$`).test(before)) return "class";
  if (new RegExp(`^\\s*(?:async\\s+)?def\\s+$`).test(before)) return "function";
  if (new RegExp(`^\\s*class\\s+${name}\\b`).test(line) && line.indexOf(name) === startColumn) return "class";
  if (new RegExp(`^\\s*(?:async\\s+)?def\\s+${name}\\b`).test(line) && line.indexOf(name) === startColumn) return "function";
  return null;
}

function safeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function markdownCode(value: string): string {
  return value.replace(/`/g, "\\`");
}

function colorHint(color: string | undefined): string | null {
  if (!color) return null;
  const safe = safeHtml(color);
  return `<span style="display:inline-block;width:0.8em;height:0.8em;border-radius:999px;border:1px solid #7f8da3;background:${safe};vertical-align:-0.1em;margin-right:0.35em;"></span><code>${safe}</code>`;
}

function symbolMarkdown(symbol: CompletionSymbol, title = symbol.name): { value: string; supportHtml: boolean } {
  const lines = [`**${safeHtml(title)}**`];
  if (symbol.detail) lines.push(`\`${markdownCode(symbol.detail)}\``);
  const color = colorHint(symbol.color);
  if (color) lines.push(color);
  if (symbol.doc) lines.push(symbol.doc);
  return { value: lines.join("\n\n"), supportHtml: true };
}

function memberMarkdown(member: PythonMember): { value: string; supportHtml: boolean } {
  const lines = [`**${safeHtml(member.name)}**`];
  if (member.detail || member.kind) lines.push(`\`${markdownCode(member.detail || member.kind)}\``);
  if (member.doc) lines.push(member.doc);
  return { value: lines.join("\n\n"), supportHtml: true };
}

function symbolByName(symbols: CompletionSymbol[] | undefined, name: string): CompletionSymbol | undefined {
  return (symbols ?? []).find((symbol) =>
    symbol.name === name || symbol.insert === name || symbol.name === `"${name}"` || symbol.insert === `"${name}"`,
  );
}

function attributeSymbols(symbols: CompletionSymbol[] | undefined): CompletionSymbol[] {
  return (symbols ?? []).filter((symbol) => symbol.access !== "item");
}

function itemSymbols(symbols: CompletionSymbol[] | undefined): CompletionSymbol[] {
  return (symbols ?? []).filter((symbol) => symbol.access === "item");
}

function bracketTarget(linePrefix: string): { target: string; quote: string | null } | null {
  const quoted = linePrefix.match(/([A-Za-z_][A-Za-z0-9_]*)\[\s*(["'])[^"']*$/);
  if (quoted) return { target: quoted[1], quote: quoted[2] };
  const unquoted = linePrefix.match(/([A-Za-z_][A-Za-z0-9_]*)\[\s*$/);
  if (unquoted) return { target: unquoted[1], quote: null };
  return null;
}

function itemHoverMarkdownForPosition(
  model: any,
  position: any,
  activeMembers: Record<string, CompletionSymbol[]> | undefined,
): { value: string; supportHtml: boolean } | null {
  const line = String(model.getLineContent(position.lineNumber));
  const pattern = /([A-Za-z_][A-Za-z0-9_]*)\[\s*(["'])([^"']+)\2\s*\]/g;
  for (const match of line.matchAll(pattern)) {
    const start = match.index ?? 0;
    const end = start + match[0].length;
    if (position.column - 1 < start || position.column - 1 > end) continue;
    const [, target, , key] = match;
    const symbol = symbolByName(itemSymbols(activeMembers?.[target]), key);
    return symbol ? symbolMarkdown(symbol, `${target}[${JSON.stringify(key)}]`) : null;
  }
  return null;
}

function hoverMarkdownForPosition(
  model: any,
  position: any,
  completionContext: CompletionContext,
  catalog: Map<string, ExpressionInfo>,
  aliases: Map<string, string>,
): { value: string; supportHtml: boolean } | null {
  const line = String(model.getLineContent(position.lineNumber));
  const word = model.getWordAtPosition(position);
  const wordText = word?.word ?? "";
  const wordStart = word ? word.startColumn - 1 : position.column - 1;
  const prefix = line.slice(0, wordStart);

  const expression = expressionEndingAt(prefix.trimEnd().replace(/\.$/, ""));
  if (expression && wordText) {
    const symbol = symbolByName(resolveExpressionMembers(expression, catalog, aliases), wordText);
    if (symbol) return symbolMarkdown(symbol, `${expression}.${wordText}`);
  }

  for (const [candidate, info] of catalog) {
    const index = line.indexOf(candidate);
    if (index >= 0 && position.column - 1 >= index && position.column - 1 <= index + candidate.length) {
      return info.symbol ? symbolMarkdown(info.symbol, candidate) : null;
    }
  }

  const aliasedExpression = wordText ? aliases.get(wordText) : undefined;
  const aliasInfo = aliasedExpression ? catalog.get(aliasedExpression) : undefined;
  if (aliasInfo?.symbol) {
    return symbolMarkdown(aliasInfo.symbol, `${wordText}: ${aliasedExpression}`);
  }

  return null;
}

function pythonIdentifier(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9_]/g, "_");
  return /^[A-Za-z_]/.test(cleaned) ? cleaned : `_${cleaned}`;
}

function replaceEvery(source: string, search: string, replacement: string): string {
  return source.split(search).join(replacement);
}

function appendSetupBlock(existing: string, block: string): string {
  const seenImports = new Set(
    existing
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.startsWith("import ") || line.startsWith("from ")),
  );
  const lines = block
    .trim()
    .split("\n")
    .filter((line) => {
      const trimmed = line.trim();
      if (!(trimmed.startsWith("import ") || trimmed.startsWith("from "))) return true;
      if (seenImports.has(trimmed)) return false;
      seenImports.add(trimmed);
      return true;
    });
  const next = lines.join("\n").trim();
  if (!next) return existing;
  return `${existing.trimEnd()}${existing.trim() ? "\n\n" : ""}${next}\n`;
}

function specializeTaskInfoScaffold(type: TaskInfoType, providerName: string): { setup: string; body: string } {
  const scaffold = type.scaffold;
  if (!scaffold) return { setup: "", body: TASK_INFO_BODY_TEMPLATE };
  const sourceName = scaffold.name;
  const targetName = pythonIdentifier(providerName);
  const targetVar = `${targetName.toUpperCase()}_TASK_INFO_PROVIDER`;
  return {
    setup: replaceEvery(
      replaceEvery(scaffold.setup, scaffold.provider_var, targetVar),
      `${sourceName}_`,
      `${targetName}_`,
    ),
    body: replaceEvery(scaffold.body, scaffold.provider_var, targetVar),
  };
}

function isTaskInfoProviderReference(entry: TaskInfoEntry | null): boolean {
  const body = entry?.body.trim() ?? "";
  return /^[A-Z][A-Z0-9_]*_TASK_INFO_PROVIDER$/.test(body);
}

function taskInfoTypeForEntry(entry: TaskInfoEntry | null, types: TaskInfoType[]): TaskInfoType | undefined {
  const body = entry?.body.trim() ?? "";
  if (body.startsWith("AUTO_COST")) return types.find((type) => type.name === "AutoCostConstraintTaskInfoProvider");
  if (body.startsWith("CONSTRAINT")) return types.find((type) => type.name === "ConstraintTaskInfoProvider");
  if (body.startsWith("GOAL")) return types.find((type) => type.name === "GoalTaskInfoProvider");
  return undefined;
}

function registerCompletions(monaco: any) {
  if (completionsRegistered) return;
  completionsRegistered = true;
  const tokenLegend = {
    tokenTypes: ["namespace", "class", "function", "variable", "property", "parameter"],
    tokenModifiers: [],
  };
  const semanticEmitter = new monaco.Emitter();
  semanticTokensDidChange = semanticEmitter;
  monaco.editor.defineTheme("vs-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [],
    colors: {},
    semanticHighlighting: true,
    semanticTokenColors: {
      namespace: "#4EC9B0",
      class: "#4EC9B0",
      function: "#DCDCAA",
      property: "#9CDCFE",
      parameter: "#C586C0",
      variable: "#D4D4D4",
    },
  });
  monaco.languages.registerDocumentSemanticTokensProvider("python", {
    getLegend: () => tokenLegend,
    onDidChange: semanticEmitter.event,
    provideDocumentSemanticTokens: async (model: any) => {
      const completionContext = latestCompletionContext?.ok ? latestCompletionContext : null;
      const symbols = semanticSymbolsForModel(model);
      const activeMembers = activeMembersForModel(model, completionContext);
      const scope = setupScope(editorContext(String(model.uri?.path ?? model.uri?.toString?.() ?? "")));
      const importedMembers = new Map<string, Map<string, string>>();
      await Promise.all(
        Object.entries(scope.imports).map(([alias, moduleName]) =>
          moduleMembers(moduleName)
            .then((members) => {
              importedMembers.set(
                alias,
                new Map(members.map((member) => [member.name, semanticTokenType(member.kind)])),
              );
            })
            .catch(() => undefined),
        ),
      );
      const builder = new monaco.languages.SemanticTokensBuilder(tokenLegend);
      const catalog = expressionCatalog(completionContext);
      const aliases = expressionAliasesForModel(model, catalog);
      const identifier = /\b[A-Za-z_][A-Za-z0-9_]*\b/g;
      for (let lineNumber = 1; lineNumber <= model.getLineCount(); lineNumber += 1) {
        const line = String(model.getLineContent(lineNumber));
        for (const match of line.matchAll(identifier)) {
          const startColumn = match.index ?? 0;
          const tokenType = definitionSemanticTokenType(line, startColumn, match[0]) ?? dottedSemanticTokenType(
            line,
            startColumn,
            match[0],
            activeMembers,
            importedMembers,
            catalog,
            aliases,
          ) ?? symbols.get(match[0]);
          if (!tokenType) continue;
          builder.push(lineNumber - 1, startColumn, match[0].length, tokenType, []);
        }
      }
      return builder.build();
    },
    releaseDocumentSemanticTokens: () => undefined,
  });
  monaco.languages.registerCompletionItemProvider("python", {
    triggerCharacters: ['"', "'", "."],
    provideCompletionItems: async (model: any, position: any) => {
      const completionContext = latestCompletionContext?.ok ? latestCompletionContext : null;
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      const K = monaco.languages.CompletionItemKind;
      const items: any[] = [];
      const add = (label: string, detail: string, kind: number, insert?: string) =>
        items.push({ label, kind, insertText: insert ?? label, detail, range });
      const addSymbol = (symbol: CompletionSymbol) => {
        items.push({
          label: symbol.name,
          kind: completionKind(symbol.kind, K),
          insertText: symbol.insert ?? symbol.name,
          detail: symbol.detail ?? "",
          documentation: symbolMarkdown(symbol),
          range,
        });
      };
      const addMember = (member: PythonMember) => {
        items.push({
          label: member.name,
          kind: completionKind(member.kind, K),
          insertText: member.name,
          detail: member.detail || member.kind,
          documentation: memberMarkdown(member),
          range,
        });
      };

      const path = String(model.uri?.path ?? model.uri?.toString?.() ?? "");
      const context = editorContext(path);
      const scope = setupScope(context);
      const hookName = context === "hook" ? hookNameFromPath(path) : null;
      const activeHookContext = hookName ? completionContext?.hooks?.[hookName] : undefined;
      const activeMembers = context === "task_info"
        ? completionContext?.task_info?.members
        : activeHookContext?.members;
      const linePrefix = String(model.getLineContent(position.lineNumber)).slice(0, position.column - 1);
      const itemTarget = bracketTarget(linePrefix);
      if (itemTarget && activeMembers?.[itemTarget.target]) {
        itemSymbols(activeMembers[itemTarget.target]).forEach((symbol) => {
          addSymbol({
            ...symbol,
            insert: itemTarget.quote ? symbol.name : `"${symbol.name}"`,
          });
        });
        return { suggestions: items };
      }

      const catalog = expressionCatalog(completionContext);
      const aliases = expressionAliasesBefore(model, position, catalog);
      const expression = completionExpression(linePrefix);
      if (completionContext && expression) {
        const members = resolveExpressionMembers(expression, catalog, aliases);
        if (members.length > 0) {
          members.forEach(addSymbol);
          return { suggestions: items };
        }
      }

      const memberTarget = linePrefix.match(/([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]*$/)?.[1]
        ?? linePrefix.match(/([A-Za-z_][A-Za-z0-9_]*)\.$/)?.[1];
      if (memberTarget && scope.imports[memberTarget]) {
        const moduleName = scope.imports[memberTarget];
        const members = await moduleMembers(moduleName);
        members.forEach(addMember);
        return { suggestions: items };
      }
      if (memberTarget && activeMembers?.[memberTarget]) {
        attributeSymbols(activeMembers[memberTarget]).forEach(addSymbol);
        return { suggestions: items };
      }
      if (memberTarget) {
        return { suggestions: items };
      }

      if (context === "custom_code") {
        [
          "ObsField", "PairObsField", "ActionField", "ObsMeta", "ActionMeta",
          "ObsQuantity", "Unit", "ControlAxis", "ActionMode",
          "MinMaxNormalizer", "SymmetricNormalizer", "CircularNormalizer", "RawNormalizer",
        ].forEach((name) => add(name, "bluesky-sandbox field API", K.Class));
        [
          "Unit.DEG", "Unit.FT", "Unit.KTS", "Unit.NM", "Unit.UNITLESS", "Unit.SWITCH",
          "ObsQuantity.LATITUDE", "ObsQuantity.LONGITUDE", "ObsQuantity.ALTITUDE",
          "ObsQuantity.SPEED", "ObsQuantity.DISTANCE", "ObsQuantity.HEADING",
          "ControlAxis.HEADING", "ControlAxis.SPEED", "ControlAxis.ALTITUDE",
          "ActionMode.ABSOLUTE", "ActionMode.DELTA", "ActionMode.SWITCH",
        ].forEach((name) => add(name, "metadata enum", K.EnumMember));
        [
          "bs.traf.lat[idx]", "bs.traf.lon[idx]", "bs.traf.alt[idx] / ft",
          "bs.traf.cas[idx] / kts", "bs.traf.hdg[idx]", "bs.traf.trk[idx]",
          "bs.traf.id[idx]", "self._configured_bounds()",
        ].forEach((expr) => add(expr, "custom field helper", K.Value));

      }

      if (context !== "custom_code" && context !== "other") {
        const importNames = new Set(Object.keys(scope.imports));
        scope.symbols.forEach((symbol) => addSymbol({
          ...symbol,
          kind: importNames.has(symbol.name) ? "module" : symbol.kind,
        }));
        Object.entries(scope.imports).forEach(([name]) => {
          if (!scope.symbols.some((symbol) => symbol.name === name)) add(name, "", K.Module);
        });
      }
      if (context === "hook_setup" || context === "task_info_setup") {
        return { suggestions: items };
      }
      if (context === "hook" || context === "task_info") {
        const insideContextQuery = /context\.query(?:able)?\(\s*["'][^"']*$/.test(linePrefix);
        if (completionContext && insideContextQuery) {
          (completionContext.queryables ?? []).forEach(addSymbol);
        }
      }
      if (context === "hook" || context === "task_info") {
        const params = context === "task_info"
          ? completionContext?.task_info?.params
          : activeHookContext?.params;
        (params ?? []).forEach(addSymbol);
      }
      if (context === "hook" && activeMembers?.self) {
        addSymbol({ name: "self", kind: "variable", detail: "environment instance" });
      }
      return { suggestions: items };
    },
  });
  monaco.languages.registerHoverProvider("python", {
    provideHover: (model: any, position: any) => {
      const completionContext = latestCompletionContext?.ok ? latestCompletionContext : null;
      if (!completionContext) return null;
      const catalog = expressionCatalog(completionContext);
      const contextHover = hoverMarkdownForPosition(
        model,
        position,
        completionContext,
        catalog,
        expressionAliasesBefore(model, position, catalog),
      );
      if (contextHover) return { contents: [contextHover] };

      const word = model.getWordAtPosition(position)?.word;
      const path = String(model.uri?.path ?? model.uri?.toString?.() ?? "");
      const context = editorContext(path);
      const hookName = context === "hook" ? hookNameFromPath(path) : null;
      const activeHookContext = hookName ? completionContext.hooks?.[hookName] : undefined;
      const activeMembers = context === "task_info"
        ? completionContext.task_info?.members
        : activeHookContext?.members;
      const itemHover = itemHoverMarkdownForPosition(model, position, activeMembers);
      if (itemHover) return { contents: [itemHover] };

      if (!word) return null;
      const scope = setupScope(context);
      const scopeSymbol = symbolByName(scope.symbols, word);
      if (scopeSymbol) return { contents: [symbolMarkdown(scopeSymbol)] };

      const activeParams = context === "task_info"
        ? completionContext.task_info?.params
        : activeHookContext?.params;
      const param = symbolByName(activeParams, word);
      if (param) return { contents: [symbolMarkdown(param)] };
      return null;
    },
  });
}

// VS Code-like view of the task's *code structure* (not raw JSON). Editable
// helper modules such as custom_fields.py live in spec.code and are edited here
// as Python; the rest of the package (scenario/env/__main__) is generated
// read-only so you can see the full structure. A "spec.json" entry keeps the
// raw document available for power edits.
const SPEC_FILE = "spec.json";

function langOf(path: string): string {
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".json")) return "json";
  if (path.endsWith(".md")) return "markdown";
  return "plaintext";
}

export default function CodeTab({
  spec,
  specText,
  onSpecChange,
  onSpecTextChange,
  validation,
}: {
  spec: SpecDict | null;
  specText: string;
  onSpecChange: (next: SpecDict) => void;
  onSpecTextChange: (text: string) => void;
  validation: ValidateResult | null;
}) {
  const [generated, setGenerated] = useState<Record<string, string>>({});
  const [pkg, setPkg] = useState<string>("");
  const [selected, setSelected] = useState<string>("design.py");
  const [genError, setGenError] = useState<string | null>(null);
  const [hookCatalog, setHookCatalog] = useState<any[]>([]);
  // Scenario hooks come from the backend (catalog.scenario_hooks, derived from
  // spec.SCENARIO_HOOKS) rather than a list duplicated here, so adding a hook
  // server-side surfaces it in the editor with no frontend change.
  const [scenarioHookCatalog, setScenarioHookCatalog] = useState<any[]>([]);
  const [taskInfoTypes, setTaskInfoTypes] = useState<TaskInfoType[]>([]);

  useEffect(() => {
    api
      .catalogOnce()
      .then((c) => {
        setHookCatalog(c?.hooks ?? []);
        setScenarioHookCatalog(c?.scenario_hooks ?? []);
        setTaskInfoTypes(c?.task_info_types ?? []);
      })
      .catch(() => {
        setHookCatalog([]);
        setScenarioHookCatalog([]);
        setTaskInfoTypes([]);
      });
  }, []);

  useEffect(() => {
    if (!spec) {
      latestCompletionContext = null;
      return;
    }
    let cancelled = false;
    const handle = setTimeout(() => {
      api
        .completions(spec)
        .then((ctx) => {
          if (!cancelled) {
            latestCompletionContext = ctx;
            semanticTokensDidChange?.fire?.();
          }
        })
        .catch((error) => {
          if (!cancelled) {
            latestCompletionContext = { ok: false, error: String(error) };
            semanticTokensDidChange?.fire?.();
          }
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [spec]);

  const codeFiles = useMemo(() => Object.keys(spec?.code ?? {}), [spec]);
  const hooks: Record<string, string> = spec?.env?.hooks ?? {};
  const hookSetup: string = spec?.env?.hook_setup ?? "";
  const taskInfoSetup: string = spec?.env?.task_info_setup ?? "";
  const taskInfo: TaskInfoEntry[] = spec?.env?.task_info ?? [];
  const externalTaskInfoProviders: string[] = spec?.env?.task_info_providers ?? [];
  // reward/terminated/truncated always exist (default hooks); shown first and
  // not deletable. Other customised hooks follow and can be removed.
  const DEFAULT_HOOKS = ["reward", "terminated", "truncated"];
  const DEFAULT_BODIES: Record<string, string> = {
    reward: "return 0.0",
    terminated: "return False",
    truncated: "return False",
  };
  const shownHooks = [...DEFAULT_HOOKS, ...Object.keys(hooks).filter((n) => !DEFAULT_HOOKS.includes(n))];
  const selectedHookSetup = selected === "hooksetup";
  const selectedHook = selected.startsWith("hook:") ? selected.slice(5) : null;
  const selectedTaskInfoSetup = selected === "taskinfo:setup";
  const selectedTaskInfoText = selected.startsWith("taskinfo:") ? selected.slice(9) : null;
  const selectedTaskInfo = selectedTaskInfoText && /^\d+$/.test(selectedTaskInfoText)
    ? Number(selectedTaskInfoText)
    : null;
  const selectedTaskInfoEntry =
    selectedTaskInfo != null && Number.isInteger(selectedTaskInfo)
      ? taskInfo[selectedTaskInfo]
      : null;
  const selectedTaskInfoProviderReference = isTaskInfoProviderReference(selectedTaskInfoEntry);
  const selectedTaskInfoType = taskInfoTypeForEntry(selectedTaskInfoEntry, taskInfoTypes);
  const hookMeta = (name: string) => hookCatalog.find((h) => h.name === name);
  const hookBody = (name: string) => hooks[name] ?? DEFAULT_BODIES[name] ?? "";

  const scaffoldHook = (meta: any): string => {
    // Prefer a curated, hook-specific scaffold when the backend provides one.
    if (meta?.scaffold) return `${meta.scaffold}\n`;
    const doc = meta?.doc ? `# ${meta.doc}\n` : "";
    const body = meta?.returns_none
      ? "return None"
      : `return ${meta?.default ?? "None"}  # TODO: return ${meta?.returns || "the value this hook expects"}`;
    return `${doc}${body}\n`;
  };
  const setHook = (name: string, body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, env: { ...spec.env, hooks: { ...hooks, [name]: body } } });
  };
  const removeHook = (name: string) => {
    if (!spec || DEFAULT_HOOKS.includes(name)) return;
    const next = { ...hooks };
    delete next[name];
    onSpecChange({ ...spec, env: { ...spec.env, hooks: next } });
    if (selected === `hook:${name}`) setSelected(`hook:reward`);
  };
  const addHook = (name: string) => {
    if (!name || hooks[name]) return;
    setHook(name, scaffoldHook(hookMeta(name)));
    setSelected(`hook:${name}`);
  };

  const setHookSetup = (body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, env: { ...spec.env, hook_setup: body } });
  };

  // Scenario code lives at the top level of the spec, not under env: it is
  // emitted into scenario.py and also compiled by the builder for live preview.
  const scenarioSetup: string = spec?.scenario_setup ?? "";
  const scenarioHooks: Record<string, string> = spec?.scenario_hooks ?? {};
  const selectedScenarioSetup = selected === "scenariosetup";
  const selectedScenarioHook = selected.startsWith("scenariohook:")
    ? selected.slice("scenariohook:".length)
    : null;
  const scenarioHookMeta = (name: string) => scenarioHookCatalog.find((h) => h.name === name);
  const setScenarioSetup = (body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, scenario_setup: body });
  };
  const setScenarioHook = (name: string, body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, scenario_hooks: { ...scenarioHooks, [name]: body } });
  };
  const removeScenarioHook = (name: string) => {
    if (!spec) return;
    const next = { ...scenarioHooks };
    delete next[name];
    onSpecChange({ ...spec, scenario_hooks: next });
    if (selected === `scenariohook:${name}`) setSelected(SPEC_FILE);
  };
  const addScenarioHook = (name: string) => {
    if (!name || scenarioHooks[name]) return;
    const meta = scenarioHookMeta(name);
    setScenarioHook(name, meta?.scaffold ?? "return geometry\n");
    setSelected(`scenariohook:${name}`);
  };

  const setTaskInfoSetup = (body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, env: { ...spec.env, task_info_setup: body } });
  };

  const setTaskInfoBody = (index: number, body: string) => {
    if (!spec) return;
    const next = [...taskInfo];
    if (!next[index]) return;
    next[index] = { ...next[index], body };
    onSpecChange({ ...spec, env: { ...spec.env, task_info: next } });
  };

  const removeTaskInfoProvider = (index: number) => {
    if (!spec) return;
    const next = taskInfo.filter((_, i) => i !== index);
    onSpecChange({ ...spec, env: { ...spec.env, task_info: next } });
    if (selected === `taskinfo:${index}`) setSelected(SPEC_FILE);
  };

  const uniqueTaskInfoName = (base: string): string => {
    const used = new Set(taskInfo.map((entry) => entry.name));
    if (!used.has(base)) return base;
    let i = 2;
    while (used.has(`${base}_${i}`)) i += 1;
    return `${base}_${i}`;
  };

  const addTaskInfoProvider = (typeName: string) => {
    if (!spec) return;
    const taskInfoType = taskInfoTypes.find((type) => type.name === typeName);
    const baseName = taskInfoType?.scaffold?.name ?? "task_info";
    const providerName = uniqueTaskInfoName(baseName);
    const scaffold = taskInfoType ? specializeTaskInfoScaffold(taskInfoType, providerName) : { setup: "", body: TASK_INFO_BODY_TEMPLATE };
    const nextProviders = [
      ...taskInfo,
      {
        name: providerName,
        body: scaffold.body,
      },
    ];
    onSpecChange({
      ...spec,
      env: {
        ...spec.env,
        task_info_setup: appendSetupBlock(taskInfoSetup, scaffold.setup),
        task_info: nextProviders,
      },
    });
    setSelected(taskInfoType ? "taskinfo:setup" : `taskinfo:${nextProviders.length - 1}`);
  };

  // Regenerate the structural files when the (valid) spec settles.
  useEffect(() => {
    if (!spec) return;
    const name = spec.metadata?.name || "designed_task";
    const handle = setTimeout(() => {
      api
        .generate(spec, name)
        .then((r) => {
          setPkg(r.package);
          setGenerated(r.files);
          setGenError(null);
        })
        .catch((e) => setGenError(String(e)));
    }, 500);
    return () => clearTimeout(handle);
  }, [spec]);

  // Generated files that are NOT editable design code (those come from spec.code).
  const structuralFiles = useMemo(() => {
    const prefix = pkg ? `${pkg}/` : "";
    return Object.keys(generated)
      .filter((p) => {
        const base = p.startsWith(prefix) ? p.slice(prefix.length) : p;
        return !codeFiles.includes(base);
      })
      .sort();
  }, [generated, codeFiles, pkg]);

  const addFile = () => {
    const name = window.prompt("new module filename", "custom_fields.py");
    if (!name || !name.endsWith(".py") || !spec) return;
    const initial = name === "custom_fields.py" ? CUSTOM_FIELDS_TEMPLATE : `# ${name}\n`;
    onSpecChange({ ...spec, code: { ...spec.code, [name]: initial } });
    setSelected(name);
  };

  const updateCode = (file: string, value: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, code: { ...spec.code, [file]: value } });
  };

  const deleteCode = (file: string) => {
    if (!spec) return;
    const next = { ...spec.code };
    delete next[file];
    onSpecChange({ ...spec, code: next });
    if (selected === file) setSelected(SPEC_FILE);
  };

  const isSpec = selected === SPEC_FILE;
  const isCode = codeFiles.includes(selected);
  const structuralKey = pkg && structuralFiles.includes(`${pkg}/${selected}`) ? `${pkg}/${selected}` : selected;
  const value = selectedHook
    ? hookBody(selectedHook)
    : selectedHookSetup
      ? hookSetup
    : selectedTaskInfoSetup
      ? taskInfoSetup
    : selectedTaskInfoEntry != null
      ? selectedTaskInfoEntry.body
    : selectedScenarioSetup
      ? scenarioSetup
    : selectedScenarioHook
      ? (scenarioHooks[selectedScenarioHook] ?? "")
    : isSpec
      ? specText
      : isCode
        ? (spec?.code?.[selected] ?? "")
        : (generated[structuralKey] ?? generated[selected] ?? "");
  const editorPath = selectedHook
    ? `hook_${selectedHook}.py`
    : selectedHookSetup
      ? "hook_setup.py"
      : selectedTaskInfoEntry
        ? `task_info_${selectedTaskInfoEntry.name}.py`
      : selectedTaskInfoSetup
        ? "task_info_setup.py"
      : selectedScenarioSetup
        ? "scenario_setup.py"
      : selectedScenarioHook
        ? `scenario_${selectedScenarioHook}.py`
      : selected;
  const editorLanguage =
    selectedHook
    || selectedHookSetup
    || selectedTaskInfoEntry
    || selectedTaskInfoSetup
    || selectedScenarioSetup
    || selectedScenarioHook
      ? "python"
      : langOf(selected);
  const editorReadOnly =
    !isSpec
    && !isCode
    && !selectedHook
    && !selectedHookSetup
    && !selectedTaskInfoSetup
    && !selectedScenarioSetup
    && !selectedScenarioHook
    && (selectedTaskInfoEntry == null || selectedTaskInfoProviderReference);

  return (
    <div className="code-tab">
      <aside className="file-tree">
        <div className="tree-group">design</div>
        <FileItem path={SPEC_FILE} active={isSpec} onClick={() => setSelected(SPEC_FILE)} />

        <div className="tree-group">
          code <button className="link" onClick={addFile}>+ file</button>
        </div>
        {codeFiles.map((f) => (
          <FileItem
            key={f}
            path={f}
            active={selected === f}
            editable
            onClick={() => setSelected(f)}
            onDelete={() => deleteCode(f)}
          />
        ))}

        <div className="tree-group">task info</div>
        <FileItem
          path="setup"
          active={selectedTaskInfoSetup}
          editable
          onClick={() => setSelected("taskinfo:setup")}
        />
        <Picker
          className="hook-add"
          placeholder="+ task info…"
          title="add task-info provider"
          onChange={addTaskInfoProvider}
          options={[
            { value: "custom", label: "custom", description: "Free-form task diagnostics provider.", category: "custom" },
            ...taskInfoTypes.map((type) => ({
              value: type.name,
              label: type.name,
              description: type.doc,
              category: type.category ?? "task info",
            })),
          ]}
        />
        {taskInfo.map((provider, i) => (
          <FileItem
            key={`${provider.name}-${i}`}
            path={provider.name}
            active={selected === `taskinfo:${i}`}
            editable
            onClick={() => setSelected(`taskinfo:${i}`)}
            onDelete={() => removeTaskInfoProvider(i)}
          />
        ))}
        {taskInfo.length === 0 && <div className="muted small file-empty">none</div>}
        {externalTaskInfoProviders.length > 0 && (
          <>
            <div className="tree-group">task info imports</div>
            {externalTaskInfoProviders.map((ref, i) => (
              <FileItem
                key={`${ref}-${i}`}
                path={ref}
                active={false}
                onClick={() => setSelected(SPEC_FILE)}
              />
            ))}
          </>
        )}

        <div className="tree-group">env hooks</div>
        <FileItem
          path="setup"
          active={selectedHookSetup}
          editable
          onClick={() => setSelected("hooksetup")}
        />
        {shownHooks.map((h) => (
          <FileItem
            key={h}
            path={h}
            active={selected === `hook:${h}`}
            editable
            onClick={() => setSelected(`hook:${h}`)}
            onDelete={DEFAULT_HOOKS.includes(h) ? undefined : () => removeHook(h)}
          />
        ))}
        {hookCatalog.length > 0 && (
          <Picker
            className="hook-add"
            placeholder="+ override hook…"
            title="override an environment hook"
            onChange={addHook}
            options={hookCatalog
              .filter((h) => !DEFAULT_HOOKS.includes(h.name) && !hooks[h.name])
              .map((h) => ({
                value: h.name,
                label: h.name,
                description: h.doc,
                category: h.category ?? "other",
              }))}
          />
        )}

        <div className="tree-group">scenario code</div>
        <FileItem
          path="setup"
          active={selectedScenarioSetup}
          editable
          onClick={() => setSelected("scenariosetup")}
        />
        {Object.keys(scenarioHooks).map((h) => (
          <FileItem
            key={h}
            path={h}
            active={selected === `scenariohook:${h}`}
            editable
            onClick={() => setSelected(`scenariohook:${h}`)}
            onDelete={() => removeScenarioHook(h)}
          />
        ))}
        {scenarioHookCatalog.length > 0 && (
          <Picker
            className="hook-add"
            placeholder="+ scenario hook…"
            title="sample geometry per episode from the design"
            onChange={addScenarioHook}
            options={scenarioHookCatalog
              .filter((h) => !scenarioHooks[h.name])
              .map((h) => ({
                value: h.name,
                label: h.name,
                description: h.doc,
                category: "scenario",
              }))}
          />
        )}

        <div className="tree-group">generated{pkg ? ` · ${pkg}/` : ""}</div>
        {structuralFiles.map((p) => {
          const base = pkg && p.startsWith(`${pkg}/`) ? p.slice(pkg.length + 1) : p;
          return <FileItem key={p} path={base} active={selected === base} onClick={() => setSelected(base)} />;
        })}
        {genError && <div className="error-text small">{genError}</div>}
      </aside>

      <div className="editor-pane">
        {selectedHook && (
          <div className="hook-sig small">
            <code className="hook-sig-def">
              def <span className="hook-sig-name">{selectedHook}</span>
              {hookMeta(selectedHook)?.def_signature ?? "(self, …)"}:
            </code>
            {hookMeta(selectedHook)?.doc ? (
              <span className="hook-sig-doc"> — {hookMeta(selectedHook).doc}</span>
            ) : null}
          </div>
        )}
        {selectedHookSetup && (
          <div className="hook-sig small">
            <code className="hook-sig-def"># env-hook imports, constants, and helpers</code>
          </div>
        )}
        {selectedTaskInfoEntry && (
          <div className="hook-sig small">
            {selectedTaskInfoProviderReference ? (
              <code className="hook-sig-def">
                provider <span className="hook-sig-name">{selectedTaskInfoEntry.body.trim()}</span>
              </code>
            ) : (
              <code className="hook-sig-def">
                def <span className="hook-sig-name">{selectedTaskInfoEntry.name}</span>
                (obs, action, info, context, rng) -&gt; None:
              </code>
            )}
          </div>
        )}
        {selectedTaskInfoSetup && (
          <div className="hook-sig small">
            <code className="hook-sig-def"># task-info imports, constants, and helpers</code>
          </div>
        )}
        {selectedScenarioSetup && (
          <div className="hook-sig small">
            <code className="hook-sig-def">
              # scenario.py module scope: imports, constants and helpers for the scenario hooks
            </code>
          </div>
        )}
        {selectedScenarioHook && (
          <div className="hook-sig small">
            <code className="hook-sig-def">
              def _{selectedScenarioHook}
              {scenarioHookMeta(selectedScenarioHook)?.signature ?? "(geometry, rng)"}:
            </code>
            {scenarioHookMeta(selectedScenarioHook)?.doc && (
              <span className="hook-doc"> {scenarioHookMeta(selectedScenarioHook)?.doc}</span>
            )}
          </div>
        )}
        <Editor
          key={editorPath}
          height="100%"
          path={editorPath}
          language={editorLanguage}
          value={value}
          beforeMount={registerCompletions}
          onChange={(v) => {
            if (selectedHook) setHook(selectedHook, v ?? "");
            else if (selectedHookSetup) setHookSetup(v ?? "");
            else if (selectedScenarioSetup) setScenarioSetup(v ?? "");
            else if (selectedScenarioHook) setScenarioHook(selectedScenarioHook, v ?? "");
            else if (selectedTaskInfoSetup) setTaskInfoSetup(v ?? "");
            else if (selectedTaskInfoEntry != null && selectedTaskInfo != null) setTaskInfoBody(selectedTaskInfo, v ?? "");
            else if (isSpec) onSpecTextChange(v ?? "");
            else if (isCode) updateCode(selected, v ?? "");
          }}
          theme="vs-dark"
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            tabSize: isSpec ? 2 : 4,
            readOnly: editorReadOnly,
            scrollBeyondLastLine: false,
            automaticLayout: true,
            "semanticHighlighting.enabled": true,
          }}
        />
      </div>

      <aside className="inspector">
        <h3>Validation</h3>
        {!validation && <p className="muted">…</p>}
        {validation && !validation.ok && <pre className="error-text">{validation.error}</pre>}
        {validation?.ok && validation.summary && (
          <dl className="summary">
            <dt>max aircraft</dt>
            <dd>{validation.summary.max_aircraft}</dd>
            <dt>obs</dt>
            <dd>{validation.summary.obs_fields.join(", ")}</dd>
            <dt>intruder</dt>
            <dd>{validation.summary.intruder_obs_fields?.join(", ") ?? "none"}</dd>
            <dt>actions</dt>
            <dd>{validation.summary.action_fields.join(", ")}</dd>
            <dt>aircraft</dt>
            <dd>{validation.summary.allowed_aircraft.join(", ")}</dd>
            <dt>queryables</dt>
            <dd>{validation.summary.queryables.join(", ") || "none"}</dd>
          </dl>
        )}
        {selectedHook && (() => {
          const m = hookMeta(selectedHook);
          return (
            <div className="hook-hint">
              <h3>Hook</h3>
              {m?.doc && <p className="small">{m.doc}</p>}
              <dl className="summary">
                {m?.category && (<><dt>category</dt><dd>{m.category}</dd></>)}
                {m?.params?.length ? (<><dt>params</dt><dd>{m.params.join(", ")}</dd></>) : null}
                {m?.returns && (<><dt>returns</dt><dd><code>{m.returns}</code></dd></>)}
                {m?.default != null && (<><dt>default</dt><dd><code>return {m.default}</code></dd></>)}
              </dl>
              {m?.params?.includes("context") && (
                <p className="muted small">
                  Tip: <code>context.query("name")</code> reads a queryable for this aircraft.
                </p>
              )}
            </div>
          );
        })()}
        {selectedHookSetup && (
          <div className="hook-hint">
            <h3>Hook Setup</h3>
            <p className="small">
              Module-level Python emitted before the env class. Put imports, constants, or helper functions used by hook methods here.
            </p>
          </div>
        )}
        {selectedTaskInfoEntry != null && (
          <div className="hook-hint">
            <h3>Task Info</h3>
            {selectedTaskInfoProviderReference ? (
              <>
                <p className="small">
                  Constructor-backed provider object. Edit its constructor arguments and callback functions in <code>task info/setup</code>.
                </p>
                {selectedTaskInfoType?.params?.length ? (
                  <dl className="summary">
                    <dt>type</dt>
                    <dd>{selectedTaskInfoType.name}</dd>
                    <dt>constructor</dt>
                    <dd>{selectedTaskInfoType.params.map((param: any) => param.name).join(", ")}</dd>
                  </dl>
                ) : null}
              </>
            ) : (
              <>
                <p className="small">
                  Runs once per controlled agent after observations and base info are built. Write public diagnostics under <code>info["task"]</code>.
                </p>
                <p className="muted small">
                  Useful inputs: <code>context.acid</code>, <code>context.query("name")</code>, <code>obs</code>, <code>action</code>, and <code>rng</code>.
                </p>
              </>
            )}
          </div>
        )}
        {selectedTaskInfoSetup && (
          <div className="hook-hint">
            <h3>Task Info Setup</h3>
            <p className="small">
              Module-level Python emitted before task-info providers. Put imports, constants, or helper functions here.
            </p>
          </div>
        )}
        {!isSpec && !isCode && !selectedHook && !selectedHookSetup && selectedTaskInfoEntry == null && !selectedTaskInfoSetup && <p className="muted small">generated · read-only</p>}
      </aside>
    </div>
  );
}

function FileItem({
  path,
  active,
  editable,
  onClick,
  onDelete,
}: {
  path: string;
  active: boolean;
  editable?: boolean;
  onClick: () => void;
  onDelete?: () => void;
}) {
  return (
    <div className={active ? "file-item active" : "file-item"} onClick={onClick}>
      <span className="file-name">
        {editable ? "✎ " : ""}
        {path}
      </span>
      {onDelete && (
        <button
          className="chip-x"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
        >
          ✕
        </button>
      )}
    </div>
  );
}
