// Rewrites GitHub-style relative links between docs pages (e.g.
// `../agent-features/compaction.md#anchor`) into site routes
// (e.g. `/CCA/docs/agent-features/compaction/#anchor`), so the same Markdown
// works both on GitHub (docs/public/) and on the Starlight site.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DOCS_ROOT = fs.realpathSync(
	fileURLToPath(new URL('../../../docs/public/', import.meta.url)),
);
const MD_LINK = /^([^:#?]+\.md)(#.*)?$/;

export default function remarkMdLinks({ base = '/', prefix = 'docs/' } = {}) {
	const basePath = base.endsWith('/') ? base : `${base}/`;

	const rewrite = (node, fileDir) => {
		if (node.type === 'link' || node.type === 'definition') {
			const match = MD_LINK.exec(node.url ?? '');
			if (match && !match[1].startsWith('/')) {
				const target = path.resolve(fileDir, match[1]);
				const rel = path.relative(DOCS_ROOT, target);
				if (!rel.startsWith('..') && !path.isAbsolute(rel)) {
					const slug = rel
						.split(path.sep)
						.join('/')
						.replace(/\.md$/, '')
						.replace(/(^|\/)index$/, '');
					const route = slug ? `${slug}/` : '';
					node.url = `${basePath}${prefix}${route}${match[2] ?? ''}`;
				}
			}
		}
		for (const child of node.children ?? []) rewrite(child, fileDir);
	};

	return (tree, file) => {
		if (!file.path) return;
		const fileDir = path.dirname(fs.realpathSync(file.path));
		if (path.relative(DOCS_ROOT, fileDir).startsWith('..')) return;
		rewrite(tree, fileDir);
	};
}
