"""Trace observed generation dependencies; these are not causal attributions."""


def generation_lineage(node_id, nodes):
    rows = {n['id'] if isinstance(n, dict) else n.id: n if isinstance(n, dict) else vars(n) for n in nodes}
    roots, missing, derived, visiting, visited = {}, set(), set(), set(), set()

    def visit(ident):
        if ident in visiting:
            missing.add(ident)
            return
        if ident in visited:
            return
        row = rows.get(ident)
        if row is None:
            missing.add(ident)
            return
        visiting.add(ident)
        mode, origin = row.get('generation_mode'), row.get('origin', 'generate')
        if origin in {'improve', 'fe_improve', 'replay', 'cross_fe_replay'}:
            derived.add(ident)
            refs = set(row.get('feedback_node_ids', []) + row.get('context_node_ids', []))
            refs.update(r for r in (row.get('based_on'), row.get('reference_mt_id')) if r is not None)
            if not refs:
                missing.add(ident)
            for ref in sorted(refs):
                visit(ref)
        elif mode in {'free', 'experience'}:
            roots[ident] = {'node_id': ident, 'stage': row.get('stage'), 'mode': mode,
                            'experience_source': row.get('experience_source')}
            if mode == 'experience' and not row.get('experience_source'):
                missing.add(ident)
        else:
            missing.add(ident)
        # A referenced MT also depends on its executed FE representation.
        parent = row.get('parent')
        parent_id = parent if isinstance(parent, int) else getattr(parent, 'id', None)
        if row.get('stage') == 'MT' and parent_id is not None:
            visit(parent_id)
        visiting.remove(ident)
        visited.add(ident)

    visit(node_id)
    modes = {r['mode'] for r in roots.values()}
    kind = ('unknown' if missing else 'mixed' if len(modes) > 1 else
            'experience' if modes == {'experience'} else 'free' if modes == {'free'} else 'unknown')
    return {'kind': kind, 'derived': bool(derived), 'roots': list(roots.values()),
            'derived_node_ids': sorted(derived), 'unknown_node_ids': sorted(missing),
            'causal_attribution': False}
