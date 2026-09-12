import { useId, useMemo } from 'react';
import { ALL_FEATURES, ACTIVATION_LABELS, type DatasetId } from '@nn-playground/engine';
import { PREPARED_PRESETS, MAX_NEURONS_PER_LAYER } from '@nn-playground/shared';
import type { RecipeDraftController, SetupTab } from '../../../hooks/useRecipeDraft.ts';
import { deriveDatasetPreviewModel } from '../../controls/datasetPreviewModel.ts';
import { DatasetPreviewCanvas } from '../../controls/DatasetPreviewCanvas.tsx';
import './setup.css';
export type { SetupTab } from '../../../hooks/useRecipeDraft.ts';
const DATASETS: [DatasetId, string][] = [
    ['circle', 'Circle'],
    ['xor', 'XOR'],
    ['gauss', 'Gaussian'],
    ['spiral', 'Spiral'],
    ['moons', 'Moons'],
    ['checkerboard', 'Checker'],
    ['rings', 'Rings'],
    ['heart', 'Heart'],
    ['three-class-clusters', 'Three-Class'],
    ['reg-plane', 'Plane'],
    ['reg-gauss', 'Multi-Gauss']];
const OPTIMIZERS = { sgd: { kind: 'sgd' }, 'sgd-momentum': { kind: 'sgd-momentum', momentum: 0.9 }, adam: { kind: 'adam', beta1: 0.9, beta2: 0.999, epsilon: 1e-8 } };
const SCHEDULES = { constant: { kind: 'constant' }, step: { kind: 'step', interval: 100, gamma: 0.9 }, cosine: { kind: 'cosine', totalSteps: 1000, minimumRate: 0.001 } };
const LABELS: Record<string, string> = {
    ...ACTIVATION_LABELS,
    sgd: 'Stochastic gradient descent', 'sgd-momentum': 'SGD with momentum', adam: 'Adam',
    xavier: 'Xavier', he: 'He', uniform: 'Uniform random', zeros: 'All zeros',
    constant: 'Constant', step: 'Step decay', cosine: 'Cosine decay',
    'mean-squared-error': 'Mean squared error', huber: 'Huber',
    'binary-cross-entropy-with-logits': 'Binary cross-entropy (logits)',
    'categorical-cross-entropy-with-logits': 'Categorical cross-entropy (logits)',
    none: 'None', l1: 'L1 (absolute weights)', l2: 'L2 (squared weights)',
    'global-norm': 'Global gradient norm',
    'binary-classification': 'Binary classification',
    'multiclass-classification': 'Multiclass classification', regression: 'Regression',
};
export function SetupEditor({ controller: c, tab, onTabChange }: { controller: RecipeDraftController; tab: SetupTab; onTabChange(tab: SetupTab): void; }) {
    const fieldPrefix = useId();
    const r = c.recipe;
    const previewSeed = r?.data.seed;
    const previewNoise = r?.data.noise;
    const previews = useMemo(() => DATASETS.map(([datasetId, label]) => ({
        datasetId, label,
        model: Number.isInteger(previewSeed) && previewSeed! >= 0 && previewSeed! <= 4294967295
            && Number.isFinite(previewNoise) && previewNoise! >= 0 && previewNoise! <= 100
            ? deriveDatasetPreviewModel({ datasetId, seed: previewSeed!, noise: previewNoise! }) : null,
    })), [previewSeed, previewNoise]);
    const selectedPreview = useMemo(() => {
        if (!r || !Number.isInteger(r.data.seed) || r.data.seed < 0 || r.data.seed > 4294967295 || !Number.isFinite(r.data.noise) || r.data.noise < 0 || r.data.noise > 100 || !Number.isInteger(r.data.sampleCount) || r.data.sampleCount < 2 || r.data.sampleCount > 1000) return null;
        return deriveDatasetPreviewModel({ datasetId: r.task.dataset, seed: r.data.seed, noise: r.data.noise, sampleCount: r.data.sampleCount });
    }, [r]);
    if (!r) return <p role="alert">No validated experiment is available for setup.</p>;
    const fieldErrors = (path: string) => c.issues
        .filter((issue) => issue.startsWith(`recipe.${path.replace(/\.(\d+)/g, '[$1]')}:`))
        .map((issue) => issue.slice(issue.indexOf(':') + 1).trim());
    const number = (label: string, path: string) => {
        const errors = fieldErrors(path);
        const errorId = `${fieldPrefix}-${path}-error`;
        return <div className="atelier-field">
            <label htmlFor={`${fieldPrefix}-${path}`}>{label}</label>
            <input id={`${fieldPrefix}-${path}`} type="text" inputMode="decimal" value={String(c.value(path) ?? '')}
                aria-invalid={errors.length > 0 || undefined}
                aria-describedby={errors.length ? errorId : undefined}
                onChange={(event) => c.commands.number(path, event.target.value)} />
            {errors.length > 0 && <span id={errorId} className="atelier-field-error">
                {errors.join(' ')}
            </span>}
        </div>;
    };
    const select = (label: string, path: string, options: readonly string[], change?: (value: string) => void) =>
        <label className="atelier-field">
            {label}
            <select value={String(c.value(path))}
                onChange={(event) => change ? change(event.target.value) : c.commands.set(path, event.target.value)}>
                {options.map((value) => <option key={value} value={value}>
                    {LABELS[value] ?? value}
                </option>)}
            </select>
        </label>;
    return <section className="atelier-setup" aria-label="Experiment setup" aria-busy={c.busy}>
        <nav className="atelier-setup-tabs" aria-label="Setup sections">
            {(['dataset', 'network', 'training'] as const).map((t) => <button key={t} type="button" aria-current={tab === t ? 'page' : undefined} onClick={() => onTabChange(t)}>
                {t === 'dataset' ? 'Data' : t === 'network' ? 'Inputs & layers' : 'Training'}
            </button>)}
        </nav>
        <fieldset disabled={c.busy} className="atelier-setup-fields">
            <details className="atelier-recipe-presets">
                <summary>Start from a preset</summary>
                <p>A preset replaces the whole draft. Review or adjust it, then apply all changes together.</p>
                <div className="atelier-preset-choices" role="group" aria-label="Recipe presets">
                    {PREPARED_PRESETS.map((preset) => <button type="button" key={preset.id}
                        aria-pressed={JSON.stringify(r) === JSON.stringify(preset.prepared.document.recipe)}
                        onClick={() => c.commands.preset(preset.prepared.document.recipe)}>
                        <strong>
                            {preset.title}
                        </strong>{' '}
                        <span>
                            {preset.description}
                        </span>
                    </button>)}
                </div>
            </details>
            {tab === 'dataset' && <div className="atelier-setup-columns">
                <section className="atelier-section">
                    <h2>Choose a pattern</h2>
                    <p>Select a dataset to generate and experiment with.</p>
                    <div className="atelier-datasets">
                        {previews.map(({ datasetId, label, model }) => <button type="button" key={datasetId} aria-pressed={r.task.dataset === datasetId} onClick={() => c.commands.dataset(datasetId)}>
                            {model ? <DatasetPreviewCanvas model={model} /> : <span>Preview unavailable</span>}
                            <span>
                                {label}
                            </span>
                        </button>)}
                    </div>
                </section>
                <section className="atelier-section">
                    <h2>
                        {DATASETS.find(([id]) => id === r.task.dataset)?.[1]}
                    </h2>
                    <p>
                        {LABELS[r.task.kind]} · draft preview</p>
                    {selectedPreview && <div className="atelier-selected-preview">
                        <DatasetPreviewCanvas model={selectedPreview} />
                    </div>}
                    <div className="atelier-form-grid">
                        {number('Samples', 'data.sampleCount')}
                        <div className="atelier-sample-presets" role="group" aria-label="Sample count presets">
                            {[100, 300, 600, 1000].map((count) => <button type="button" key={count} aria-pressed={r.data.sampleCount === count} onClick={() => c.commands.number('data.sampleCount', String(count))}>
                                {count}
                            </button>)}
                        </div>
                        {number('Noise (%)', 'data.noise')}
                        {number('Train fraction (0.1–0.9)', 'data.trainFraction')}
                        {number('Data seed', 'data.seed')}
                    </div>
                    <p>
                        {Number.isFinite(r.data.sampleCount * r.data.trainFraction) ? `${Math.floor(r.data.sampleCount * r.data.trainFraction)} training · ${r.data.sampleCount - Math.floor(r.data.sampleCount * r.data.trainFraction)} test samples` : 'Complete the sample count and split to preview the population.'}
                    </p>
                    <button type="button" disabled={!Number.isInteger(r.data.seed) || r.data.seed >= 4294967295} onClick={() => c.commands.number('data.seed', String(r.data.seed + 1))}>Reshuffle seed</button>
                </section>
            </div>}
            {tab === 'network' && <>
                <h2>Shape the network</h2>
                <div className="atelier-network-draft" role="group" tabIndex={0} aria-label="Draft architecture">
                    <div>
                        <h3>Input</h3>
                        <p>
                            {r.inputs.featureIds.length} features</p><div className="atelier-draft-inputs">{r.inputs.featureIds.map((id) => <span key={id}>{ALL_FEATURES.find((feature) => feature.id === id)?.label ?? id}</span>)}</div>
                    </div>
                    {r.model.hiddenLayers.map((width, i) => <section key={i}>
                        <h3>Hidden {String(i + 1).padStart(2, '0')}
                        </h3>
                        <div className="atelier-neuron-symbols" aria-hidden="true">
                            {Array.from({ length: Number.isFinite(width) ? Math.max(0, Math.min(16, width)) : 0 }, (_, j) => <span key={j}>{j + 1}</span>)}
                        </div>
                        <div className="atelier-neuron-stepper"><button type="button" aria-label={`Decrease layer ${i + 1} neurons`} disabled={!Number.isInteger(width) || width <= 1} onClick={() => c.commands.number(`model.hiddenLayers.${i}`, String(width - 1))}>−</button>{number(`Layer ${i + 1} neurons`, `model.hiddenLayers.${i}`)}<button type="button" aria-label={`Increase layer ${i + 1} neurons`} disabled={!Number.isInteger(width) || width >= MAX_NEURONS_PER_LAYER} onClick={() => c.commands.number(`model.hiddenLayers.${i}`, String(width + 1))}>+</button></div>
                        <button type="button" onClick={() => c.commands.set('model.hiddenLayers', r.model.hiddenLayers.filter((_, j) => i !== j))}>Remove layer {i + 1}
                        </button>
                    </section>)}
                    <div>
                        <h3>Output</h3>
                        <p>
                            {r.task.kind === 'multiclass-classification' ? '3 outputs · softmax' : r.task.kind === 'regression' ? '1 output · linear' : '1 output · sigmoid'}
                        </p>
                    </div>
                    <button type="button" disabled={r.model.hiddenLayers.length >= 6} onClick={() => c.commands.set('model.hiddenLayers', [...r.model.hiddenLayers, 4])}>+ Add hidden layer</button>
                </div>
                <div className="atelier-setup-columns">
                    <section className="atelier-section">
                        <h3>Input features</h3>
                        <p>Transform inputs to make patterns easier to separate.</p>
                        <div className="atelier-feature-choices">
                            {ALL_FEATURES.map((f) => <label key={f.id}>
                                <input type="checkbox" checked={r.inputs.featureIds.includes(f.id)} onChange={() => c.commands.set('inputs.featureIds', ALL_FEATURES.filter((candidate) => candidate.id === f.id ? !r.inputs.featureIds.includes(f.id) : r.inputs.featureIds.includes(candidate.id)).map((candidate) => candidate.id))} />
                                {f.label}
                            </label>)}
                        </div>
                    </section>
                    <section className="atelier-section atelier-form-grid">
                        {select('Hidden activation', 'model.hiddenActivation', Object.keys(ACTIVATION_LABELS).filter((v) => v !== 'softmax'))}{select('Weight initialization', 'model.initialization', ['xavier', 'he', 'uniform', 'zeros'])}
                        {number('Model seed', 'model.seed')}
                        <p>Output shape and activation follow the selected dataset.</p>
                    </section>
                </div>
            </>}
            {tab === 'training' && <>
                <h2>Shape the learning process</h2>
                <p>Configure how the model learns from data.</p>
                <div className="atelier-setup-columns">
                    <div>
                        <section className="atelier-section">
                            <h3>Updates</h3>
                            <div className="atelier-form-grid">
                                {number('Learning rate', 'training.learningRate')}
                                {number('Batch size', 'training.batchSize')}{select('Optimizer', 'training.optimizer.kind', Object.keys(OPTIMIZERS), (v) => c.commands.set('training.optimizer', OPTIMIZERS[v as keyof typeof OPTIMIZERS]))}
                                {r.training.optimizer.kind === 'sgd-momentum' && number('Momentum', 'training.optimizer.momentum')}
                                {r.training.optimizer.kind === 'adam' && <>
                                    {number('Beta 1', 'training.optimizer.beta1')}
                                    {number('Beta 2', 'training.optimizer.beta2')}
                                    {number('Epsilon', 'training.optimizer.epsilon')}
                                </>}
                            </div>
                        </section>
                        <section className="atelier-section">
                            <h3>Learning-rate schedule</h3>
                            {select('Schedule', 'training.schedule.kind', Object.keys(SCHEDULES), (v) => c.commands.set('training.schedule', SCHEDULES[v as keyof typeof SCHEDULES]))}
                            <div className="atelier-form-grid">
                                {r.training.schedule.kind === 'step' && <>
                                    {number('Interval (steps)', 'training.schedule.interval')}
                                    {number('Gamma', 'training.schedule.gamma')}
                                </>}
                                {r.training.schedule.kind === 'cosine' && <>
                                    {number('Duration (steps)', 'training.schedule.totalSteps')}
                                    {number('Minimum rate', 'training.schedule.minimumRate')}
                                </>}
                            </div>
                        </section>
                    </div>
                    <div>
                        <section className="atelier-section">
                            <h3>Objective</h3>
                            {r.task.kind === 'regression' ? <>
                                {select('Data loss', 'objective.dataLoss.kind', ['mean-squared-error', 'huber'], (v) => c.commands.set('objective.dataLoss', v === 'huber' ? { kind: v, delta: 1 } : { kind: v }))}
                                {r.objective.dataLoss.kind === 'huber' && number('Huber delta', 'objective.dataLoss.delta')}
                            </> : <p>
                                {LABELS[r.objective.dataLoss.kind]}
                            </p>}
                            <p>Mean per sample.</p>
                        </section>
                        <section className="atelier-section">
                            <h3>Regularization</h3>
                            {select('Penalty', 'objective.penalty.kind', ['none', 'l1', 'l2'], (v) => c.commands.set('objective.penalty', v === 'none' ? { kind: v } : { kind: v, coefficient: 0.001, applyTo: 'weights' }))}
                            {r.objective.penalty.kind !== 'none' && number('Coefficient', 'objective.penalty.coefficient')}
                            <p>Applies to weights only.</p>
                        </section>
                        <section className="atelier-section">
                            <h3>Gradient clipping</h3>
                            {select('Clipping', 'training.gradientClipping.kind', ['none', 'global-norm'], (v) => c.commands.set('training.gradientClipping', v === 'none' ? { kind: v } : { kind: v, maximumNorm: 1, scope: 'total-objective-gradient' }))}
                            {r.training.gradientClipping.kind === 'global-norm' && number('Maximum norm', 'training.gradientClipping.maximumNorm')}
                            <p>Clips the total objective gradient.</p>
                        </section>
                    </div>
                </div>
            </>}
        </fieldset>
        {c.issues.length > 0 && <div role="alert" className="atelier-setup-errors">
            <p>Complete these settings before applying:</p>
            <ul>
                {c.issues.map((issue) => <li key={issue}>
                    {issue}
                </li>)}
            </ul>
        </div>}
        {c.error && <div role="alert">
            <p>
                {c.error}
            </p>
            {c.submitted && <button type="button" onClick={c.commands.retry}>Retry setup sync</button>}
        </div>}
        <footer className="atelier-setup-footer">
            <p>
                {c.submitted ? 'Waiting for the new experiment to be ready…' : 'Applying setup restarts training. Changes across all three sections apply together.'}
            </p>
            <button type="button" disabled={c.busy || !c.dirty} onClick={c.commands.cancel}>Cancel</button>
            <button type="button" className="atelier-action" disabled={!c.dirty || !c.valid || c.busy} onClick={() => void c.commands.apply()}>Apply changes</button>
        </footer>
    </section>;
}
