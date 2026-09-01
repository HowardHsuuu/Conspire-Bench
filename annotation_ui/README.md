# Local blinded annotation UI

This static interface loads the public JSONL files produced by `analysis/export_annotations.py`. It never loads `annotation_key.private.jsonl`, has no external dependencies, and makes no network requests.

## Use

1. Export a pilot or formal annotation package.
2. Open `annotation_ui/index.html` in a current browser. If the browser restricts local-file APIs, serve the repository locally with `python3 -m http.server 8000` and open `http://localhost:8000/annotation_ui/`.
3. Load the assigned expert or student JSONL file. If it was produced by `analysis/assign_annotations.py`, the interface reads and locks the pseudonymous annotator ID; pilot master files require the ID to be entered manually.
4. Complete ratings. Progress is kept in that browser's local storage under the package-content digest and annotator ID.
5. Export completed JSONL and return it to the research team. The output is accepted directly by `analysis/import_annotations.py`.

Do not distribute the private key, frozen source results, model/frame labels, or automated judge scores to annotators. Establish the study's ethics, consent, compensation, withdrawal, distress-support, device, and data-retention procedures before formal collection.
