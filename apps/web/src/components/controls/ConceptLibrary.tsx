import { CONCEPTS } from '../../concepts/conceptCatalog.ts';
import { useLayoutStore } from '../../store/useLayoutStore.ts';

/** Canonical vocabulary remains reachable at every guidance level and screen size. */
export function ConceptLibrary() {
    const guidance = useLayoutStore((state) => state.audienceMode);
    return <section className="atelier-concepts" aria-label="Concept library">
        <h3>Understand the experiment</h3>
        {CONCEPTS.map((entry) => <details key={entry.id}>
            <summary>{entry.canonicalTerm}</summary>
            <div><p>{entry.plainDefinition}</p>
                {guidance !== 'lab' && entry.extendedExplanation && <p>{entry.extendedExplanation}</p>}
                {guidance === 'beginner' && entry.examples?.map((example) => <p key={example}>Example: {example}</p>)}
                <p>Related: {entry.related.map((id) => CONCEPTS.find((concept) => concept.id === id)?.canonicalTerm).join(', ')}</p>
            </div>
        </details>)}
    </section>;
}
