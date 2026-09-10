"""Static checks for the two bugs that keep reaching Actions: a name that no longer exists
(load_manifest, _MB_BY_MODEL), and a function whose returns disagree on arity. ast.parse sees
neither, and both take the build down on the first run."""
import ast, builtins, sys, types, importlib.util

for m in ["pygrib","requests","matplotlib","matplotlib.pyplot","matplotlib.colors",
          "matplotlib.patheffects","cartopy","cartopy.crs","cartopy.feature"]:
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["matplotlib"].use = lambda *a, **k: None

PATH = sys.argv[1] if len(sys.argv) > 1 else "cloudscope.py"
spec = importlib.util.spec_from_file_location("mod", PATH)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
tree = ast.parse(open(PATH).read())
missing, arity = [], []


def binds(node):
    """Names this scope binds, not descending into nested scopes."""
    out = set()
    def tgt(t):
        if isinstance(t, ast.Name): out.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts: tgt(e)
        elif isinstance(t, ast.Starred): tgt(t.value)
    args = getattr(node, "args", None)
    if args:
        for a in list(args.args) + list(args.kwonlyargs) + list(args.posonlyargs):
            out.add(a.arg)
        if args.vararg: out.add(args.vararg.arg)
        if args.kwarg: out.add(args.kwarg.arg)
    def walk(n, top=True):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(c.name)          # the name is bound here; its body is its own scope
                continue
            if isinstance(c, ast.Lambda):
                continue
            if isinstance(c, ast.Assign):
                for t in c.targets: tgt(t)
            elif isinstance(c, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                tgt(c.target)
            elif isinstance(c, (ast.For, ast.AsyncFor, ast.comprehension)):
                tgt(c.target)
            elif isinstance(c, ast.withitem) and c.optional_vars is not None:
                tgt(c.optional_vars)
            elif isinstance(c, ast.ExceptHandler) and c.name:
                out.add(c.name)
            elif isinstance(c, (ast.Global, ast.Nonlocal)):
                out.update(c.names)
            elif isinstance(c, (ast.Import, ast.ImportFrom)):
                out.update((a.asname or a.name.split(".")[0]) for a in c.names)
            walk(c, False)
    walk(node)
    return out


def loads(node):
    """Name loads in this scope, not descending into nested scopes."""
    out = []
    def walk(n):
        for c in ast.iter_child_nodes(n):
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                out.append(c)
            walk(c)
    walk(node)
    return out


def visit(node, enclosing, name):
    scope = enclosing | binds(node)
    for n in loads(node):
        if n.id not in scope and not hasattr(mod, n.id) and not hasattr(builtins, n.id):
            missing.append((name, n.id, n.lineno))
    if isinstance(node, ast.FunctionDef):
        ars = {len(r.value.elts) if isinstance(r.value, ast.Tuple) else 1
               for r in [x for x in ast.walk(node) if isinstance(x, ast.Return)]
               if r.value is not None
               and all(r not in ast.walk(f) for f in ast.iter_child_nodes(node)
                       if isinstance(f, ast.FunctionDef))}
        if len(ars) > 1:
            arity.append((name, sorted(ars)))
    for c in ast.iter_child_nodes(node):
        if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            visit(c, scope, getattr(c, "name", "<lambda>"))
        else:
            for d in ast.walk(c):
                if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    visit(d, scope, getattr(d, "name", "<lambda>"))


for c in ast.iter_child_nodes(tree):
    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
        visit(c, set(), c.name)

for fname, n, line in sorted(set(missing)):
    print(f"  UNDEFINED  {n}  in {fname}() line {line}")
for fname, ars in arity:
    print(f"  ARITY      {fname}() returns {ars}")
print(f"undefined: {len(set(missing))}  arity mismatches: {len(arity)}")
sys.exit(1 if (missing or arity) else 0)
