// What code in a design can use, and how an expression resolves against it.
//
// The backend (ui/designer/code_intel.py) describes types, each code block's
// scope, and the keys the design fixes - all by introspection. This resolves an
// expression chain - `context.raw_obs["intruders"]["acid"]`, `goal.current` after
// `goal = context.query("goal")` - one step at a time against that description.
// Nothing here knows a member by name.

export type Param = {
  name: string;
  detail?: string;
  type?: string;
  default?: string;
  // The argument is one of a design's keys: the type whose items they are.
  keys?: string;
};

export type Member = {
  name: string;
  kind: string;
  detail?: string;
  doc?: string;
  // The value's type - for a function, what it returns.
  type?: string;
  params?: Param[];
  // A call with a key argument returns that key's item type.
  returns_by_key?: string;
  // For a class: the instance type its call returns.
  returns?: string;
  module?: string;
  color?: string;
};

// `closed`: the items are all there are - a design's keys.
export type TypeInfo = { name: string; doc?: string; attrs: Member[]; items?: Member[]; closed?: boolean };

// A problem the backend found in a block, 1-based in the block's own text.
export type Problem = { line: number; column: number; end_column?: number; message: string; severity: string };

export type Scope = { params: Member[]; names: string };

export type Intel = {
  ok: true;
  scopes: Record<string, Scope>;
  names: Record<string, Member[]>;
  types: Record<string, TypeInfo>;
};

// ---------------------------------------------------------------- tokens --
export type Token =
  | { kind: "name"; text: string; start: number; end: number }
  | { kind: "string"; text: string; value: string; closed: boolean; start: number; end: number }
  | { kind: "punct"; text: string; start: number; end: number }
  | { kind: "other"; text: string; start: number; end: number };

// Python-ish tokens of one line; a string left open at the end is marked so.
export function tokenize(line: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < line.length) {
    const c = line[i];
    if (c === "#") break;
    if (/\s/.test(c)) {
      i += 1;
      continue;
    }
    const prefix = line.slice(i).match(/^([rRbBfFuU]{0,2})(["'])/);
    if (prefix) {
      const quote = prefix[2];
      let j = i + prefix[0].length;
      let value = "";
      let closed = false;
      while (j < line.length) {
        if (line[j] === "\\") {
          value += line.slice(j, j + 2);
          j += 2;
          continue;
        }
        if (line[j] === quote) {
          closed = true;
          j += 1;
          break;
        }
        value += line[j];
        j += 1;
      }
      tokens.push({ kind: "string", text: line.slice(i, j), value, closed, start: i, end: j });
      i = j;
      continue;
    }
    const name = line.slice(i).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (name) {
      tokens.push({ kind: "name", text: name[0], start: i, end: i + name[0].length });
      i += name[0].length;
      continue;
    }
    if ("()[]{}.,:=".includes(c)) {
      tokens.push({ kind: "punct", text: c, start: i, end: i + 1 });
    } else {
      const other = line.slice(i).match(/^[^\sA-Za-z_"'()[\]{}.,:=#]+/);
      const text = other ? other[0] : c;
      tokens.push({ kind: "other", text, start: i, end: i + text.length });
      i += text.length;
      continue;
    }
    i += 1;
  }
  return tokens;
}

// ----------------------------------------------------------------- chains --
// `root .name ["key"] (args)` - an access chain, left to right.
export type Step =
  | { op: "attr"; name: string; token: Token }
  | { op: "item"; key: string | null; token: Token }
  | { op: "call"; key: string | null; token: Token };

export type Chain = { root: Token; steps: Step[] };

// The chain ending at token index `end` (exclusive), read backwards; null if
// the tokens there are not one.
export function chainBefore(tokens: Token[], end: number): Chain | null {
  const steps: Step[] = [];
  let i = end - 1;
  while (i >= 0) {
    const t = tokens[i];
    if (t.kind === "name") {
      if (i > 0 && tokens[i - 1].text === ".") {
        steps.unshift({ op: "attr", name: t.text, token: t });
        i -= 2;
        continue;
      }
      return { root: t, steps };
    }
    if (t.text === "]" || t.text === ")") {
      const open = matchingOpen(tokens, i);
      if (open < 0) return null;
      const inner = tokens.slice(open + 1, i);
      const key = inner[0]?.kind === "string" ? (inner[0] as any).value : null;
      const literal = inner.length === 1 && inner[0].kind === "string" ? key : null;
      if (t.text === "]") steps.unshift({ op: "item", key: literal, token: inner[0] ?? t });
      else steps.unshift({ op: "call", key, token: inner[0] ?? t });
      i = open - 1;
      continue;
    }
    return null;
  }
  return null;
}

function matchingOpen(tokens: Token[], close: number): number {
  const pairs: Record<string, string> = { ")": "(", "]": "[", "}": "{" };
  const stack: string[] = [];
  for (let i = close; i >= 0; i -= 1) {
    const text = tokens[i].text;
    if (tokens[i].kind !== "punct") continue;
    if (text in pairs) stack.push(pairs[text]);
    else if (text === "(" || text === "[" || text === "{") {
      if (stack.pop() !== text) return -1;
      if (stack.length === 0) return i;
    }
  }
  return -1;
}

// ------------------------------------------------------------- resolution --
export type Resolved = { member?: Member; type?: string; module?: string };

export class Resolver {
  private aliases = new Map<string, { tokens: Token[]; line: number }>();
  private resolving = new Set<string>();

  constructor(
    readonly intel: Intel,
    readonly scope: Scope | undefined,
    lines: string[],
  ) {
    // `name = <chain>` lines give a name the chain's type.
    lines.forEach((line, n) => {
      const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?!=)(.+)$/);
      if (match) this.aliases.set(match[1], { tokens: tokenize(match[2]), line: n });
    });
  }

  names(): Member[] {
    if (!this.scope) return [];
    return [...this.scope.params, ...(this.intel.names[this.scope.names] ?? [])];
  }

  type(key: string | undefined): TypeInfo | undefined {
    return key ? this.intel.types[key] : undefined;
  }

  root(name: string): Resolved | null {
    const member = this.names().find((m) => m.name === name);
    if (member) return { member, type: member.type, module: member.module };
    const alias = this.aliases.get(name);
    if (alias && !this.resolving.has(name)) {
      this.resolving.add(name);
      try {
        const chain = chainBefore(alias.tokens, alias.tokens.length);
        const resolved = chain ? this.resolve(chain) : null;
        if (resolved) return { ...resolved, member: { name, kind: "variable", detail: resolved.member?.detail, doc: resolved.member?.doc, type: resolved.type } };
      } finally {
        this.resolving.delete(name);
      }
    }
    return null;
  }

  isAlias(name: string): boolean {
    return this.aliases.has(name);
  }

  // Each step's result, left to right; stops where a step does not resolve.
  walk(chain: Chain): Resolved[] {
    const out: Resolved[] = [];
    let current = this.root(chain.root.text);
    if (!current) return out;
    out.push(current);
    for (const step of chain.steps) {
      current = this.step(current, step);
      if (!current) break;
      out.push(current);
    }
    return out;
  }

  resolve(chain: Chain): Resolved | null {
    const walked = this.walk(chain);
    return walked.length === chain.steps.length + 1 ? walked[walked.length - 1] : null;
  }

  step(from: Resolved, step: Step): Resolved | null {
    if (step.op === "attr") {
      const member = this.type(from.type)?.attrs.find((m) => m.name === step.name);
      return member ? { member, type: member.type, module: member.module } : null;
    }
    if (step.op === "item") {
      if (step.key === null) return null;
      const member = this.type(from.type)?.items?.find((m) => m.name === step.key);
      return member ? { member, type: member.type } : null;
    }
    const callee = from.member;
    if (!callee) return null;
    if (callee.returns_by_key && step.key !== null) {
      const item = this.type(callee.returns_by_key)?.items?.find((m) => m.name === step.key);
      return item ? { member: item, type: item.type } : null;
    }
    const type = callee.kind === "class" ? callee.returns : callee.type;
    return type ? { member: { ...callee, kind: "value" }, type } : null;
  }

  // What the argument of a call may be, when the design fixes it.
  argumentKeys(callee: Resolved | null, index: number): Member[] {
    const param = callee?.member?.params?.[index];
    return (param?.keys && this.type(param.keys)?.items) || [];
  }
}

// ----------------------------------------------------------------- scopes --
// The scope of an editor model, from its path - the paths the Code tab and the
// field modal give their editors.
export function scopeKey(path: string): string | null {
  const file = path.split("/").pop() ?? path;
  if (file === "hook_setup.py") return "hook_setup";
  if (file === "task_info_setup.py") return "task_info_setup";
  if (file === "scenario_setup.py") return "scenario_setup";
  const hook = file.match(/^hook_(.+)\.py$/);
  if (hook) return `hook:${hook[1]}`;
  if (/^task_info_.+\.py$/.test(file)) return "task_info";
  const scenario = file.match(/^scenario_(.+)\.py$/);
  if (scenario) return `scenario:${scenario[1]}`;
  const field = path.match(/custom_field_([^:/]+):/);
  if (field) return `code:${field[1]}.py`;
  return file.endsWith(".py") ? `code:${file}` : null;
}

// The block an editor model shows - as the backend's diagnostics key it. The
// scope's key, except that each task-info entry is its own block.
export function blockKey(path: string): string | null {
  const file = path.split("/").pop() ?? path;
  const entry = file.match(/^task_info_(.+)\.py$/);
  if (entry && file !== "task_info_setup.py") return `task_info:${entry[1]}`;
  // The field modal shows one class of a module: the module's lines do not apply.
  if (/custom_field_/.test(path)) return null;
  return scopeKey(path);
}

// The names nearest to `text`, for a "did you mean" hint.
export function nearest(text: string, names: string[], limit = 2): string[] {
  const distance = (a: string, b: string) => {
    const row = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i += 1) {
      let previous = row[0];
      row[0] = i;
      for (let j = 1; j <= b.length; j += 1) {
        const current = row[j];
        row[j] = Math.min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] === b[j - 1] ? 0 : 1));
        previous = current;
      }
    }
    return row[b.length];
  };
  const budget = Math.max(2, Math.floor(text.length / 3));
  return names
    .map((name) => [name, distance(text, name)] as const)
    .filter(([, d]) => d <= budget)
    .sort((a, b) => a[1] - b[1])
    .slice(0, limit)
    .map(([name]) => name);
}

export function memberMarkdown(member: Member, title: string): { value: string; supportHtml: boolean } {
  const lines = [`**${escapeHtml(title)}**`];
  if (member.detail) lines.push("`" + member.detail.replace(/`/g, "\\`") + "`");
  if (member.color) {
    const color = escapeHtml(member.color);
    lines.push(
      `<span style="display:inline-block;width:0.8em;height:0.8em;border-radius:999px;` +
        `border:1px solid #7f8da3;background:${color};vertical-align:-0.1em;` +
        `margin-right:0.35em;"></span><code>${color}</code>`,
    );
  }
  if (member.doc) lines.push(member.doc);
  return { value: lines.join("\n\n"), supportHtml: true };
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// The text of a chain, for titles: `context.raw_obs["intruders"]`.
export function chainText(chain: Chain, upTo = chain.steps.length): string {
  let text = chain.root.text;
  for (const step of chain.steps.slice(0, upTo)) {
    if (step.op === "attr") text += `.${step.name}`;
    else if (step.op === "item") text += `[${JSON.stringify(step.key)}]`;
    else text += step.key !== null ? `(${JSON.stringify(step.key)})` : "()";
  }
  return text;
}
