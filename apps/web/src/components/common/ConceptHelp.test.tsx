import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { getConceptById, type ConceptId } from '../../concepts/conceptCatalog.ts';
import { ConceptHelp } from './ConceptHelp.tsx';

describe('ConceptHelp', () => {
    it('uses a real disclosure button with a clear accessible name and relationship', async () => {
        const user = userEvent.setup();
        render(<ConceptHelp conceptId="data-loss" guidanceLevel="standard" />);

        const trigger = screen.getByRole('button', { name: 'Learn about Data loss' });
        const contentId = trigger.getAttribute('aria-controls');

        expect(trigger).toHaveAttribute('type', 'button');
        expect(trigger).toHaveAttribute('aria-expanded', 'false');
        expect(contentId).toBeTruthy();
        expect(document.getElementById(contentId ?? '')).not.toBeInTheDocument();

        await user.click(trigger);

        expect(trigger).toHaveAttribute('aria-expanded', 'true');
        expect(document.getElementById(contentId ?? '')).toBe(
            screen.getByRole('region', { name: 'Data loss' }),
        );

        await user.click(trigger);
        expect(trigger).toHaveAttribute('aria-expanded', 'false');
        expect(screen.queryByRole('region', { name: 'Data loss' })).not.toBeInTheDocument();
    });

    it('supports keyboard activation without depending on hover', async () => {
        const user = userEvent.setup();
        render(<ConceptHelp conceptId="activation" guidanceLevel="compact" />);

        const trigger = screen.getByRole('button', { name: 'Learn about Activation' });
        fireEvent.mouseEnter(trigger);
        expect(screen.queryByRole('region', { name: 'Activation' })).not.toBeInTheDocument();

        trigger.focus();
        await user.keyboard('{Enter}');
        expect(screen.getByRole('region', { name: 'Activation' })).toBeInTheDocument();

        await user.keyboard(' ');
        expect(screen.queryByRole('region', { name: 'Activation' })).not.toBeInTheDocument();
    });

    it('restores trigger focus after Escape closes the disclosure', async () => {
        const user = userEvent.setup();
        render(
            <ConceptHelp
                conceptId="data-loss"
                guidanceLevel="high"
                onNavigateToTarget={vi.fn()}
            />,
        );

        const trigger = screen.getByRole('button', { name: 'Learn about Data loss' });
        await user.click(trigger);

        const targetAction = screen.getByRole('button', { name: 'Open Loss' });
        targetAction.focus();
        expect(targetAction).toHaveFocus();

        await user.keyboard('{Escape}');

        expect(screen.queryByRole('region', { name: 'Data loss' })).not.toBeInTheDocument();
        expect(trigger).toHaveFocus();
    });

    it('always renders the plain definition and varies optional detail by guidance level', async () => {
        const user = userEvent.setup();
        const entry = getConceptById('data-loss');

        const { rerender } = render(
            <ConceptHelp conceptId="data-loss" guidanceLevel="compact" />,
        );
        await user.click(screen.getByRole('button', { name: 'Learn about Data loss' }));

        expect(screen.getByText('Difficulty: Beginner')).toBeInTheDocument();
        expect(screen.getByText(entry?.plainDefinition ?? '')).toBeInTheDocument();
        expect(screen.queryByText(entry?.extendedExplanation ?? '')).not.toBeInTheDocument();
        expect(screen.queryByText(entry?.examples?.[0] ?? '')).not.toBeInTheDocument();

        rerender(<ConceptHelp conceptId="data-loss" guidanceLevel="standard" />);
        expect(screen.getByText(entry?.plainDefinition ?? '')).toBeInTheDocument();
        expect(screen.getByText(entry?.extendedExplanation ?? '')).toBeInTheDocument();
        expect(screen.queryByText(entry?.examples?.[0] ?? '')).not.toBeInTheDocument();

        rerender(<ConceptHelp conceptId="data-loss" guidanceLevel="high" />);
        expect(screen.getByText(entry?.plainDefinition ?? '')).toBeInTheDocument();
        expect(screen.getByText(entry?.extendedExplanation ?? '')).toBeInTheDocument();
        expect(screen.getByText(entry?.examples?.[0] ?? '')).toBeInTheDocument();
    });

    it('renders related concept names without duplicating their definitions', async () => {
        const user = userEvent.setup();
        render(<ConceptHelp conceptId="data-loss" guidanceLevel="high" />);
        await user.click(screen.getByRole('button', { name: 'Learn about Data loss' }));

        expect(screen.getByText('Related concepts')).toBeInTheDocument();
        expect(screen.getByText('Training objective')).toBeInTheDocument();
        expect(screen.getByText('Decision boundary')).toBeInTheDocument();
        expect(
            screen.queryByText(getConceptById('training-objective')?.plainDefinition ?? ''),
        ).not.toBeInTheDocument();
        expect(
            screen.queryByText(getConceptById('decision-boundary')?.plainDefinition ?? ''),
        ).not.toBeInTheDocument();
    });

    it('forwards the optional workspace target and omits the action when no target exists', async () => {
        const user = userEvent.setup();
        const onNavigateToTarget = vi.fn();
        const { rerender } = render(
            <ConceptHelp
                conceptId="gradient"
                guidanceLevel="standard"
                onNavigateToTarget={onNavigateToTarget}
            />,
        );

        await user.click(screen.getByRole('button', { name: 'Learn about Gradient' }));
        await user.click(screen.getByRole('button', { name: 'Open Inspect' }));
        expect(onNavigateToTarget).toHaveBeenCalledTimes(1);
        expect(onNavigateToTarget).toHaveBeenCalledWith('inspection');

        rerender(
            <ConceptHelp
                conceptId="checkpoint"
                guidanceLevel="standard"
                onNavigateToTarget={onNavigateToTarget}
            />,
        );
        expect(screen.getByRole('region', { name: 'Checkpoint' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: /^Open / })).not.toBeInTheDocument();
    });

    it('escapes the paint-contained workspace without losing disclosure focus or Escape ordering', async () => {
        const user = userEvent.setup();
        const outerEscape = vi.fn();
        const { container } = render(
            <div className="forge-shell" onKeyDown={(event) => { if (event.key === 'Escape') outerEscape(); }}>
                <div data-precision-workspace style={{ contain: 'layout paint', overflow: 'hidden' }}>
                    <ConceptHelp conceptId="data-loss" guidanceLevel="high" onNavigateToTarget={vi.fn()} />
                </div>
            </div>,
        );
        const trigger = screen.getByRole('button', { name: 'Learn about Data loss' });
        await user.click(trigger);
        const panel = screen.getByRole('region', { name: 'Data loss' });
        expect(container.querySelector('[data-precision-workspace]')).not.toContainElement(panel);
        expect(container.querySelector('.forge-shell')).toContainElement(panel);
        expect(panel).toHaveClass('concept-help__content--viewport');
        expect(panel).toHaveFocus();
        await user.tab();
        expect(screen.getByRole('button', { name: 'Open Loss' })).toHaveFocus();
        screen.getByRole('button', { name: 'Open Loss' }).focus();
        await user.keyboard('{Escape}');
        expect(panel).not.toBeInTheDocument();
        expect(trigger).toHaveFocus();
        expect(outerEscape).not.toHaveBeenCalled();
    });

    it('renders nothing for a missing concept instead of exposing an empty help control', () => {
        const { container } = render(
            <ConceptHelp
                conceptId={'missing-concept' as ConceptId}
                guidanceLevel="standard"
            />,
        );

        expect(container).toBeEmptyDOMElement();
    });
});
