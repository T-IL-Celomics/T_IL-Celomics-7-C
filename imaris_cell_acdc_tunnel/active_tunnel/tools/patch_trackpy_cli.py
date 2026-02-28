
import re
from pathlib import Path
import cellacdc.trackers.trackpy.trackpy_tracker as m

p = Path(m.__file__)
txt = p.read_text(encoding="utf-8")

bak = p.with_suffix(p.suffix + ".bak")
if not bak.exists():
    bak.write_text(txt, encoding="utf-8")

def guard_signal(src: str, signal_name: str) -> str:
    pat = re.compile(rf'^(\s*)signals\.{re.escape(signal_name)}\.emit\((.*?)\)\s*$', re.M)
    def repl(m):
        indent, args = m.group(1), m.group(2)
        return f'{indent}if hasattr(signals, "{signal_name}"):\n{indent}    signals.{signal_name}.emit({args})'
    return pat.sub(repl, src)

new = txt
new = guard_signal(new, "initProgressBar")
new = guard_signal(new, "progressBar")

if new == txt:
    print("[patch] No changes applied (maybe already patched). File:", p)
else:
    p.write_text(new, encoding="utf-8")
    print("[patch] Applied CLI-safe guards. File:", p)
    print("[patch] Backup saved as:", bak)
