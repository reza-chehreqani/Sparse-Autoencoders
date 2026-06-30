"""
Run this BEFORE run_phase1.py if you're unsure whether the sae_release /
sae_id format in config.py matches your installed sae_lens version.
The sae_lens pretrained-SAE naming scheme has changed across versions,
so this prints what's actually available so you can adjust config.py.

Run: python list_available_saes.py
"""

try:
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
except ImportError:
    print("Could not import sae_lens internals at this path.")
    print("Run `pip show sae_lens` to check your version, then consult")
    print("https://github.com/jbloomAus/SAELens for the current method")
    print("of listing pretrained SAEs (the directory module path has")
    print("moved between versions).")
    raise SystemExit(1)

directory = get_pretrained_saes_directory()
found = False
for release_name, info in directory.items():
    if "gemma-scope-2b" in release_name:
        found = True
        print(f"\nRelease: {release_name}")
        print(f"  repo_id: {info.repo_id}")
        sample_ids = list(info.saes_map.keys())[:8]
        print(f"  sample sae_ids ({len(info.saes_map)} total): {sample_ids}")

if not found:
    print("No 'gemma-scope-2b' releases found in this sae_lens version's directory.")
    print("Search the full directory manually, e.g.:")
    print("  [k for k in directory.keys() if 'gemma' in k.lower()]")
else:
    print(
        "\nCompare the sae_id pattern above against config.py's "
        "`sae_release` / `sae_width` / sae_id template "
        "(currently: f'layer_{{layer}}/{{sae_width}}/canonical'). "
        "Adjust config.py if the pattern differs."
    )
