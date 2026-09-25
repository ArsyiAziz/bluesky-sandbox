// Completion, hover, coloring and signature help for every Python editor in the
// designer, from the design's code intel (see ./intel.ts). Registered once;
// every editor - the Code tab's and the field modal's - reads the latest intel.
import { useEffect, useSyncExternalStore } from "react";
import { api, type SpecDict } from "../api";
import {
  blockKey,
  chainBefore,
  chainText,
  type Intel,
  type Member,
  memberMarkdown,
  nearest,
  type Problem,
  type Resolved,
  Resolver,
  scopeKey,
  type Token,
  tokenize,
} from "./intel";

let latest: Intel | null = null;
let problems: Record<string, Problem[]> = {};
let registered = false;
let intelChanged: any = null;
let monacoApi: any = null;
const moduleMembers = new Map<string, Promise<Member[]>>();

const listeners = new Set<() => void>();

export function currentIntel(): Intel | null {
  return latest;
}

// The latest intel, re-rendering the component when it changes.
export function useIntel(): Intel | null {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => latest,
  );
}

// Keep the intel current with the spec; editors re-color when it changes.
export function useCodeIntel(spec: SpecDict | null, refreshKey = 0) {
  useEffect(() => {
    if (!spec) return;
    let canceled = false;
    const handle = setTimeout(() => {
      api
        .codeIntel(spec)
        .then((intel) => {
          if (canceled) return;
          latest = intel?.ok ? (intel as Intel) : latest;
          intelChanged?.fire?.();
          listeners.forEach((listener) => listener());
          markAll();
        })
        .catch(() => undefined);
      api
        .diagnostics(spec)
        .then((result) => {
          if (canceled || !result?.ok) return;
          problems = result.problems;
          markAll();
        })
        .catch(() => undefined);
    }, 250);
    return () => {
      canceled = true;
      clearTimeout(handle);
    };
  }, [spec, refreshKey]);
}

/** Forget the modules' members read so far, so they are read again. */
export function forgetModuleMembers() {
  moduleMembers.clear();
}

function membersOfModule(module: string): Promise<Member[]> {
  let members = moduleMembers.get(module);
  if (!members) {
    members = api
      .pythonModuleMembers(module)
      .then((result) => result.members as Member[])
      .catch(() => {
        moduleMembers.delete(module);
        return [];
      });
    moduleMembers.set(module, members);
  }
  return members;
}

function resolverFor(model: any): Resolver | null {
  const key = scopeKey(String(model.uri?.path ?? ""));
  if (!latest || !key) return null;
  return new Resolver(latest, latest.scopes[key], String(model.getValue()).split("\n"));
}

// The semantic token type a member is colored as.
const TOKEN_TYPES = ["namespace", "class", "function", "variable", "property", "parameter"];
function tokenType(kind: string): string {
  if (kind === "module") return "namespace";
  if (kind === "class" || kind === "function" || kind === "parameter") return kind;
  if (kind === "property" || kind === "field") return "property";
  return "variable";
}

// What the chain ending before token `end` resolves to, if it is one.
function resolveBefore(resolver: Resolver, tokens: Token[], end: number): Resolved | null {
  const chain = chainBefore(tokens, end);
  return chain ? resolver.resolve(chain) : null;
}

// The key a string token stands for, when it sits in `x["key"]` or `f("key")`:
// the member it names, if the design has it.
function keyMember(resolver: Resolver, tokens: Token[], index: number): Member | null {
  const open = tokens[index - 1];
  if (!open || (open.text !== "[" && open.text !== "(")) return null;
  const value = (tokens[index] as any).value as string;
  const target = resolveBefore(resolver, tokens, index - 1);
  if (open.text === "[") return resolver.type(target?.type)?.items?.find((m) => m.name === value) ?? null;
  return resolver.argumentKeys(target, 0).find((m) => m.name === value) ?? null;
}

// The member a name token stands for: its chain, walked up to it.
function nameMember(resolver: Resolver, tokens: Token[], index: number): Resolved | null {
  const chain = chainBefore(tokens, index + 1);
  if (!chain) return null;
  return chain.steps.length === 0 ? resolver.root(chain.root.text) : resolver.resolve(chain);
}

// Semantic tokens in Monaco's encoding: five integers per token - line and start
// column, each relative to the previous token, then length, type index and
// modifier bits.
class SemanticTokens {
  private data: number[] = [];
  private line = 0;
  private column = 0;

  push(line: number, column: number, length: number, type: string) {
    const index = TOKEN_TYPES.indexOf(type);
    if (index < 0) return;
    const deltaLine = line - this.line;
    this.data.push(deltaLine, deltaLine === 0 ? column - this.column : column, length, index, 0);
    this.line = line;
    this.column = column;
  }

  build() {
    return { data: new Uint32Array(this.data) };
  }
}

// --------------------------------------------------------------- markers --
// A design key that is not one: `raw_obs["ownship"]["alt_fx"]`. Only keys the
// design fixes are checked - their sets are closed; `info["task"]` is yours.
function keyProblems(resolver: Resolver, lines: string[]): any[] {
  const markers: any[] = [];
  lines.forEach((line, n) => {
    const tokens = tokenize(line);
    tokens.forEach((token, i) => {
      if (token.kind !== "string" || !token.closed) return;
      const open = tokens[i - 1]?.text;
      if (open !== "[" && open !== "(") return;
      const target = resolveBefore(resolver, tokens, i - 1);
      let keys: Member[] | undefined;
      if (open === "[") {
        const type = resolver.type(target?.type);
        keys = type?.closed ? type.items : undefined;
      } else if (target?.member?.params?.[0]?.keys) {
        keys = resolver.argumentKeys(target, 0);
      }
      if (!keys || keys.some((m) => m.name === token.value)) return;
      const names = keys.map((m) => m.name);
      const hint = nearest(token.value, names);
      const chain = chainBefore(tokens, i - 1);
      markers.push({
        startLineNumber: n + 1,
        endLineNumber: n + 1,
        startColumn: token.start + 1,
        endColumn: token.end + 1,
        message:
          `${JSON.stringify(token.value)} is not a key of ${chain ? chainText(chain) : "this"}` +
          (hint.length ? ` - did you mean ${hint.map((h) => JSON.stringify(h)).join(" or ")}?` : ""),
        severity: monacoApi.MarkerSeverity.Error,
      });
    });
  });
  return markers;
}

function mark(model: any) {
  if (!monacoApi || model.isDisposed?.()) return;
  const path = String(model.uri?.path ?? "");
  const block = blockKey(path);
  const found = (block && problems[block]) || [];
  monacoApi.editor.setModelMarkers(
    model,
    "design",
    found.map((p) => ({
      startLineNumber: p.line,
      endLineNumber: p.line,
      startColumn: p.column,
      endColumn: p.end_column ?? model.getLineMaxColumn(Math.min(p.line, model.getLineCount())),
      message: p.message,
      severity: p.severity === "error" ? monacoApi.MarkerSeverity.Error : monacoApi.MarkerSeverity.Warning,
    })),
  );
  const resolver = resolverFor(model);
  monacoApi.editor.setModelMarkers(
    model,
    "design-keys",
    resolver ? keyProblems(resolver, String(model.getValue()).split("\n")) : [],
  );
}

function markAll() {
  monacoApi?.editor.getModels().forEach(mark);
}

export function registerPythonIntel(monaco: any) {
  if (registered) return;
  registered = true;
  monacoApi = monaco;
  intelChanged = new monaco.Emitter();
  // Keys are checked as they are typed; the backend's checks follow the spec.
  const watch = (model: any) => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    mark(model);
    model.onDidChangeContent(() => {
      clearTimeout(timer);
      timer = setTimeout(() => mark(model), 300);
    });
  };
  monaco.editor.getModels().forEach(watch);
  monaco.editor.onDidCreateModel(watch);
  const K = monaco.languages.CompletionItemKind;
  const completionKind = (kind: string) =>
    ({ module: K.Module, class: K.Class, function: K.Function, property: K.Property, field: K.Field, parameter: K.Variable, variable: K.Variable })[kind] ?? K.Value;

  // Monaco styles a semantic token through the theme rule named after its type.
  const colors: Record<string, string> = {
    namespace: "4EC9B0",
    class: "4EC9B0",
    function: "DCDCAA",
    property: "9CDCFE",
    parameter: "C586C0",
    variable: "D4D4D4",
  };
  monaco.editor.defineTheme("vs-dark", {
    base: "vs-dark",
    inherit: true,
    rules: Object.entries(colors).map(([token, foreground]) => ({ token, foreground })),
    colors: {},
    semanticHighlighting: true,
  });

  monaco.languages.registerDocumentSemanticTokensProvider("python", {
    getLegend: () => ({ tokenTypes: TOKEN_TYPES, tokenModifiers: [] }),
    onDidChange: intelChanged.event,
    provideDocumentSemanticTokens: (model: any) => {
      const resolver = resolverFor(model);
      const out = new SemanticTokens();
      if (!resolver) return out.build();
      const paramNames = new Set(resolver.scope?.params.map((p) => p.name));
      model
        .getValue()
        .split("\n")
        .forEach((line: string, n: number) => {
          const tokens = tokenize(line);
          tokens.forEach((token, i) => {
            if (token.kind === "string" && token.closed) {
              if (keyMember(resolver, tokens, i)) {
                const quote = token.text.search(/["']/);
                out.push(n, token.start + quote + 1, token.value.length, "property");
              }
              return;
            }
            if (token.kind !== "name") return;
            const before = tokens[i - 1]?.text;
            if (before === "def") return out.push(n, token.start, token.text.length, "function");
            if (before === "class") return out.push(n, token.start, token.text.length, "class");
            if (before !== "." && paramNames.has(token.text)) {
              return out.push(n, token.start, token.text.length, "parameter");
            }
            const resolved = nameMember(resolver, tokens, i);
            if (resolved?.member) out.push(n, token.start, token.text.length, tokenType(resolved.member.kind));
          });
        });
      return out.build();
    },
    releaseDocumentSemanticTokens: () => undefined,
  });

  monaco.languages.registerCompletionItemProvider("python", {
    triggerCharacters: [".", '"', "'", "["],
    provideCompletionItems: async (model: any, position: any) => {
      const resolver = resolverFor(model);
      if (!resolver) return { suggestions: [] };
      const prefix = String(model.getLineContent(position.lineNumber)).slice(0, position.column - 1);
      const tokens = tokenize(prefix);
      const word = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };
      const suggest = (members: Member[], quote = "") =>
        members.map((m) => ({
          label: m.name,
          kind: completionKind(m.kind),
          insertText: `${quote}${m.name}${quote}`,
          detail: m.detail ?? "",
          documentation: memberMarkdown(m, m.name),
          range,
        }));
      const last = tokens[tokens.length - 1];
      const before = tokens[tokens.length - 2];

      // Inside an open string: the key of `x["` or of a keyed call `f("`.
      if (last?.kind === "string" && !last.closed && before) {
        const target = resolveBefore(resolver, tokens, tokens.length - 2);
        if (before.text === "[") return { suggestions: suggest(resolver.type(target?.type)?.items ?? []) };
        if (before.text === "(") return { suggestions: suggest(resolver.argumentKeys(target, 0)) };
        return { suggestions: [] };
      }
      // Right after `[`: the keys, quoted.
      if (last?.text === "[") {
        const target = resolveBefore(resolver, tokens, tokens.length - 1);
        return { suggestions: suggest(resolver.type(target?.type)?.items ?? [], '"') };
      }
      // After a dot: the attributes of what is before it.
      const dot = last?.text === "." ? tokens.length - 1 : before?.text === "." && last?.kind === "name" ? tokens.length - 2 : -1;
      if (dot >= 0) {
        const chain = chainBefore(tokens, dot);
        const target = chain ? resolver.resolve(chain) : null;
        if (target?.module) return { suggestions: suggest(await membersOfModule(target.module)) };
        return { suggestions: suggest(resolver.type(target?.type)?.attrs ?? []) };
      }
      // A bare name: what the scope has.
      return { suggestions: suggest(resolver.names()) };
    },
  });

  monaco.languages.registerHoverProvider("python", {
    provideHover: (model: any, position: any) => {
      const resolver = resolverFor(model);
      if (!resolver) return null;
      const tokens = tokenize(String(model.getLineContent(position.lineNumber)));
      const column = position.column - 1;
      const index = tokens.findIndex((t) => t.start <= column && column < t.end);
      const token = tokens[index];
      if (!token) return null;
      if (token.kind === "string") {
        const member = keyMember(resolver, tokens, index);
        const chain = chainBefore(tokens, index - 1);
        const title = chain ? `${chainText(chain)}[${JSON.stringify((token as any).value)}]` : token.text;
        return member ? { contents: [memberMarkdown(member, title)] } : null;
      }
      if (token.kind !== "name") return null;
      const resolved = nameMember(resolver, tokens, index);
      const chain = chainBefore(tokens, index + 1);
      if (!resolved?.member || !chain) return null;
      return { contents: [memberMarkdown(resolved.member, chainText(chain))] };
    },
  });

  monaco.languages.registerSignatureHelpProvider("python", {
    signatureHelpTriggerCharacters: ["(", ","],
    signatureHelpRetriggerCharacters: [","],
    provideSignatureHelp: (model: any, position: any) => {
      const resolver = resolverFor(model);
      if (!resolver) return null;
      const prefix = String(model.getLineContent(position.lineNumber)).slice(0, position.column - 1);
      const tokens = tokenize(prefix);
      // The innermost call still open at the cursor, and the argument it is at.
      let depth = 0;
      let commas = 0;
      let open = -1;
      for (let i = tokens.length - 1; i >= 0 && open < 0; i -= 1) {
        const text = tokens[i].kind === "punct" ? tokens[i].text : "";
        if (text === ")" || text === "]" || text === "}") depth += 1;
        else if (text === "(" || text === "[" || text === "{") {
          if (depth === 0) {
            if (text !== "(") return null;
            open = i;
          } else depth -= 1;
        } else if (text === "," && depth === 0) commas += 1;
      }
      const chain = open > 0 ? chainBefore(tokens, open) : null;
      const callee = chain ? resolver.resolve(chain) : null;
      const params = callee?.member?.params;
      if (!callee?.member || !params) return null;
      const parts = params.map((p) => `${p.name}${p.detail ? `: ${p.detail}` : ""}${p.default ? ` = ${p.default}` : ""}`);
      const label = `${chainText(chain!)}(${parts.join(", ")})`;
      return {
        value: {
          signatures: [
            {
              label,
              documentation: callee.member.doc ? { value: callee.member.doc } : undefined,
              parameters: parts.map((part, i) => ({
                label: part,
                documentation: params[i].keys ? `one of: ${(resolver.argumentKeys(callee, i) ?? []).map((m) => m.name).join(", ")}` : undefined,
              })),
            },
          ],
          activeSignature: 0,
          activeParameter: Math.min(commas, Math.max(parts.length - 1, 0)),
        },
        dispose: () => undefined,
      };
    },
  });
}
