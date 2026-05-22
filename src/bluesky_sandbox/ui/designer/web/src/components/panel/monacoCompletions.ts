// Monaco completion provider for the in-modal Python editors (custom fields and
// task functions). Registered once, lazily, on first editor mount.
let modalCompletionsRegistered = false;

export function registerModalCompletions(monaco: any) {
  if (modalCompletionsRegistered) return;
  modalCompletionsRegistered = true;
  monaco.languages.registerCompletionItemProvider("python", {
    triggerCharacters: [".", "(", ","],
    provideCompletionItems: (model: any, position: any) => {
      const path = String(model.uri?.path ?? model.uri?.toString?.() ?? "");
      if (!path.includes("custom_field_")) {
        return { suggestions: [] };
      }
      const K = monaco.languages.CompletionItemKind;
      const Snip = monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet;
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: position.column,
        endColumn: position.column,
      };
      const labels = [
        "ObsField", "PairObsField", "ActionField", "ObsMeta", "ActionMeta",
        "Unit.DEG", "Unit.FT", "Unit.KTS", "Unit.NM", "Unit.UNITLESS",
        "ObsQuantity.DISTANCE", "ObsQuantity.ALTITUDE", "ObsQuantity.SPEED",
        "ControlAxis.HEADING", "ControlAxis.SPEED", "ControlAxis.ALTITUDE",
        "ActionMode.ABSOLUTE", "ActionMode.DELTA",
        "MinMaxNormalizer", "SymmetricNormalizer", "CircularNormalizer", "RawNormalizer",
        "bs.traf.lat[idx]", "bs.traf.lon[idx]", "bs.traf.alt[idx] / ft",
        "bs.traf.cas[idx] / kts", "bs.traf.hdg[idx]", "bs.stack.stack",
      ];
      const suggestions: any[] = labels.map((label) => ({
        label,
        kind: K.Value,
        insertText: label,
        detail: "custom field helper",
        range,
      }));
      suggestions.push({
        label: "bounds method",
        kind: K.Snippet,
        insertTextRules: Snip,
        insertText: "def bounds(self, idx: int):\n    return self._configured_bounds()\n",
        detail: "custom field bounds",
        range,
      });
      return { suggestions };
    },
  });
}
