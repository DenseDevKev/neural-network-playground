import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath, URL } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const web = 'apps/web/src/';
const app = `${web}App`;
const stripExtension = (path) => path.replace(/\.[cm]?[jt]sx?$/, '');

// Exact existing standalone/legacy adapters, not a general components exemption.
// Remove each exception only with its consumer-proven retirement.
const owners = new Map([
    [`${web}hooks/useTraining`, [app]],
    [`${web}hooks/useSaveCurrentRun`, [app, `${web}components/layout/MainArea`, `${web}components/controls/RunHistoryPanel`]],
    [`${web}components/visualization/useNetworkSelectionController`, [app, ...['NetworkGraph', 'NetworkGraphCanvas', 'NetworkGraphSVG'].map((name) => `${web}components/visualization/${name}`)]],
    [`${web}components/visualization/useDecisionBoundaryController`, [app]],
]);

function importTarget(filename, source) {
    if (source.startsWith('.')) return relative(root, resolve(dirname(filename), source)).replaceAll('\\', '/');
    const alias = source.match(/^@nn-playground\/(engine|shared|web)(?:\/(.*))?$/);
    if (alias) return `${alias[1] === 'web' ? 'apps' : 'packages'}/${alias[1]}/src/${alias[2] ?? 'index.ts'}`;
    return source;
}

function literal(node) {
    if (typeof node?.value === 'string') return node.value;
    if (node?.type === 'TemplateLiteral' && node.expressions.length === 0) return node.quasis[0].value.cooked;
    return null;
}

const rule = {
    meta: {
        type: 'problem',
        schema: [],
        messages: {
            domain: 'Engine/shared must not depend on application UI, stores, or UI libraries.',
            layering: 'Engine is below shared; importing shared creates a reverse domain dependency.',
            testOnly: 'Production must not import prototype/test-only code. Keep fixtures behind test entry points.',
            owner: 'Controller ownership belongs in App. Pass typed models/commands to presentation instead of importing {{source}}.',
        },
    },
    create(context) {
        const filename = context.filename;
        const importer = relative(root, filename).replaceAll('\\', '/');
        const domain = /^packages\/(engine|shared)\/src\//.test(importer);
        function check(node, sourceNode, typeOnly = false) {
            const source = literal(sourceNode);
            if (source === null) return; // Only statically resolvable imports; no runtime-count claim.
            const target = importTarget(filename, source);
            let messageId;
            if (domain && (target.startsWith('apps/web/') || /^(react|react-dom|zustand|comlink)(\/|$)/.test(target))) messageId = 'domain';
            else if (importer.startsWith('packages/engine/src/') && target.startsWith('packages/shared/')) messageId = 'layering';
            else if (/^(vitest|jest-axe|@testing-library)(\/|$)|^node:test$/.test(target) || /(^|\/)(prototypes|__tests__|__benchmarks__|test|tests)(\/|$)|\.(test|spec|bench)\.[cm]?[jt]sx?$/.test(target)) messageId = 'testOnly';
            else if (!typeOnly && owners.has(stripExtension(target)) && !owners.get(stripExtension(target)).includes(stripExtension(importer))) messageId = 'owner';
            if (messageId) context.report({ node, messageId, data: { source } });
        }
        return {
            ImportDeclaration(node) {
                check(node, node.source, node.importKind === 'type' || (node.specifiers.length > 0 && node.specifiers.every((specifier) => specifier.importKind === 'type')));
            },
            ExportNamedDeclaration(node) { if (node.source) check(node, node.source, node.exportKind === 'type' || (node.specifiers.length > 0 && node.specifiers.every((specifier) => specifier.exportKind === 'type'))); },
            ExportAllDeclaration(node) { check(node, node.source, node.exportKind === 'type'); },
            ImportExpression(node) { check(node, node.source); },
            TSImportType(node) { check(node, node.argument.type === 'TSLiteralType' ? node.argument.literal : node.argument, true); },
            CallExpression(node) {
                if (node.callee.type === 'Identifier' && node.callee.name === 'require' && node.arguments.length === 1) check(node, node.arguments[0]);
            },
        };
    },
};

export const architecturePlugin = { rules: { 'dependency-boundaries': rule } };
