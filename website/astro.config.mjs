// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';
import tailwindcss from '@tailwindcss/vite';
import remarkMdLinks from './src/plugins/remark-md-links.mjs';

export default defineConfig({
	site: 'https://cca-sh.github.io',
	base: '/CCA',
	redirects: {
		'/docs/models-and-providers/': '/CCA/docs/configuration/models-and-providers/',
		'/docs/thinking-mode/': '/CCA/docs/configuration/thinking-mode/',
		'/docs/feature-toggles/': '/CCA/docs/configuration/feature-toggles/',
		'/docs/run-modes/': '/CCA/docs/agent-features/run-modes/',
		'/docs/permissions-and-tools/': '/CCA/docs/agent-features/permissions-and-tools/',
		'/docs/sessions-and-traces/': '/CCA/docs/agent-features/sessions-and-traces/',
		'/docs/compaction/': '/CCA/docs/agent-features/compaction/',
		'/docs/memory/': '/CCA/docs/agent-features/memory/',
		'/docs/hooks/': '/CCA/docs/agent-features/hooks/',
		'/docs/skills/': '/CCA/docs/agent-features/skills/',
		'/docs/mcp/': '/CCA/docs/agent-features/mcp/',
		'/docs/web-search/': '/CCA/docs/agent-features/web-search/',
		'/docs/repl-commands/': '/CCA/docs/agent-features/repl-commands/',
		'/docs/key-bindings/': '/CCA/docs/agent-features/key-bindings/',
	},
	markdown: { remarkPlugins: [[remarkMdLinks, { base: '/CCA' }]] },
	vite: { plugins: [tailwindcss()] },
	integrations: [
		starlight({
			title: 'cca',
			customCss: ['./src/styles/theme.css'],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/CCA-sh/CCA' },
			],
			plugins: [
				starlightBlog({
					authors: {
						santaboi: { name: 'santaboi', url: 'https://github.com/santaboi' },
					},
				}),
			],
			sidebar: [
				{
					label: 'Configuration',
					items: [{ autogenerate: { directory: 'docs/configuration' } }],
				},
				{
					label: 'Agent Features',
					items: [{ autogenerate: { directory: 'docs/agent-features' } }],
				},
				{
					label: 'Tutorials',
					items: [{ autogenerate: { directory: 'docs/tutorials' } }],
				},
				{ label: 'Changelog', link: '/changelog/' },
			],
		}),
	],
});
