import { StrictMode, useRef } from 'react';
import { createPortal } from 'react-dom';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { useModalFocusContainment } from './useModalFocusContainment.ts';

interface HarnessProps {
    active: boolean;
    controls?: 'two' | 'none' | 'filtered';
    onFirstAction?: () => void;
}

function Harness({ active, controls = 'two', onFirstAction }: HarnessProps) {
    const dialogRef = useRef<HTMLDivElement>(null);
    const backgroundRef = useRef<HTMLDivElement>(null);
    useModalFocusContainment(active, dialogRef, backgroundRef);

    return (
        <>
            <div ref={backgroundRef} data-testid="background">
                <button type="button">Background opener</button>
            </div>
            {active && createPortal(
                <div
                    ref={dialogRef}
                    role="alertdialog"
                    aria-label="Fatal worker error"
                    tabIndex={-1}
                >
                    {controls !== 'none' && (
                        <>
                            <button type="button" onClick={onFirstAction}>First action</button>
                            <button type="button">Second action</button>
                        </>
                    )}
                    {controls === 'filtered' && (
                        <>
                            <button type="button" disabled>Disabled action</button>
                            <button type="button" hidden>Hidden action</button>
                            <span aria-hidden="true"><button type="button">ARIA hidden action</button></span>
                            <span inert><button type="button">Inert action</button></span>
                            <button type="button" style={{ display: 'none' }}>CSS hidden action</button>
                            <div contentEditable suppressContentEditableWarning>Editable action</div>
                            <div contentEditable suppressContentEditableWarning tabIndex={-1}>
                                Explicitly excluded editable action
                            </div>
                        </>
                    )}
                </div>,
                document.body,
            )}
        </>
    );
}

describe('useModalFocusContainment', () => {
    it('focuses the dialog and cycles a fresh enabled, visible control list in both directions', async () => {
        const user = userEvent.setup();
        const view = render(<Harness active={false} controls="filtered" />);
        const opener = screen.getByRole('button', { name: 'Background opener' });
        opener.focus();

        view.rerender(<Harness active controls="filtered" />);

        const dialog = screen.getByRole('alertdialog', { name: 'Fatal worker error' });
        const first = within(dialog).getByRole('button', { name: 'First action' });
        const second = within(dialog).getByRole('button', { name: 'Second action' });
        const editable = within(dialog).getByText('Editable action');
        expect(editable.tabIndex).toBe(-1);
        expect(dialog).toHaveFocus();
        expect(screen.getByTestId('background')).toHaveAttribute('inert');
        expect(screen.getByTestId('background')).toHaveAttribute('aria-hidden', 'true');

        await user.tab();
        expect(first).toHaveFocus();
        await user.tab();
        expect(second).toHaveFocus();
        await user.tab();
        expect(editable).toHaveFocus();
        await user.tab();
        expect(first).toHaveFocus();
        await user.tab({ shift: true });
        expect(editable).toHaveFocus();

        const inserted = document.createElement('button');
        inserted.textContent = 'Inserted action';
        dialog.append(inserted);
        await user.tab();
        expect(inserted).toHaveFocus();

        inserted.remove();
        await user.tab();
        expect(first).toHaveFocus();
    });

    it('keeps focus on the dialog when there are no eligible controls and redirects programmatic escape', async () => {
        const user = userEvent.setup();
        render(<Harness active controls="none" />);
        const dialog = screen.getByRole('alertdialog', { name: 'Fatal worker error' });

        await user.tab();
        expect(dialog).toHaveFocus();
        await user.tab({ shift: true });
        expect(dialog).toHaveFocus();

        const outside = document.createElement('button');
        outside.textContent = 'Late outside control';
        document.body.append(outside);
        try {
            outside.focus();
            expect(dialog).toHaveFocus();
        } finally {
            outside.remove();
        }
    });

    it('allows native dialog actions and modified browser recovery shortcuts', async () => {
        const user = userEvent.setup();
        const onAction = vi.fn();
        render(<Harness active onFirstAction={onAction} />);
        const action = screen.getByRole('button', { name: 'First action' });
        action.focus();

        await user.keyboard(' ');
        expect(onAction).toHaveBeenCalledTimes(1);

        for (const modifier of [{ metaKey: true }, { ctrlKey: true }]) {
            const modifiedReload = new KeyboardEvent('keydown', {
                bubbles: true,
                cancelable: true,
                code: 'KeyR',
                key: 'r',
                ...modifier,
            });
            action.dispatchEvent(modifiedReload);
            expect(modifiedReload.defaultPrevented).toBe(false);
        }
    });

    it('leases current and future body portals and restores every exact preexisting attribute value', async () => {
        const existingPortal = document.createElement('div');
        existingPortal.dataset.testid = 'existing-portal';
        existingPortal.setAttribute('inert', 'legacy-inert');
        existingPortal.setAttribute('aria-hidden', 'menu');
        document.body.append(existingPortal);

        const view = render(<Harness active={false} />);
        const background = screen.getByTestId('background');
        background.setAttribute('inert', 'legacy-background');
        background.setAttribute('aria-hidden', 'false');

        view.rerender(<Harness active />);
        expect(background).toHaveAttribute('inert', '');
        expect(background).toHaveAttribute('aria-hidden', 'true');
        expect(existingPortal).toHaveAttribute('inert', '');
        expect(existingPortal).toHaveAttribute('aria-hidden', 'true');

        const futurePortal = document.createElement('aside');
        futurePortal.setAttribute('aria-hidden', 'false');
        document.body.append(futurePortal);
        await waitFor(() => {
            expect(futurePortal).toHaveAttribute('inert', '');
            expect(futurePortal).toHaveAttribute('aria-hidden', 'true');
        });

        futurePortal.remove();
        view.rerender(<Harness active={false} />);
        expect(background).toHaveAttribute('inert', 'legacy-background');
        expect(background).toHaveAttribute('aria-hidden', 'false');
        expect(existingPortal).toHaveAttribute('inert', 'legacy-inert');
        expect(existingPortal).toHaveAttribute('aria-hidden', 'menu');
        expect(futurePortal).not.toHaveAttribute('inert');
        expect(futurePortal).toHaveAttribute('aria-hidden', 'false');

        existingPortal.remove();
    });

    it('restores leases on unmount and ignores a detached opener without throwing', () => {
        const opener = document.createElement('button');
        const bodyPortal = document.createElement('div');
        bodyPortal.setAttribute('aria-hidden', 'false');
        document.body.append(opener, bodyPortal);
        opener.focus();

        const view = render(<Harness active />);
        expect(bodyPortal).toHaveAttribute('inert');
        opener.remove();

        expect(() => view.unmount()).not.toThrow();
        expect(bodyPortal).not.toHaveAttribute('inert');
        expect(bodyPortal).toHaveAttribute('aria-hidden', 'false');
        bodyPortal.remove();
    });

    it.each([
        ['display', 'none'],
        ['visibility', 'hidden'],
        ['visibility', 'collapse'],
    ] as const)(
        'does not restore a still-connected opener inside an ancestor with %s: %s',
        (property, value) => {
            const view = render(<Harness active={false} />);
            const background = screen.getByTestId('background');
            const opener = screen.getByRole('button', { name: 'Background opener' });
            opener.focus();

            view.rerender(<Harness active />);
            background.style[property] = value;
            view.rerender(<Harness active={false} />);

            expect(opener).not.toHaveFocus();
        },
    );

    it('survives StrictMode setup-cleanup-setup without capturing the dialog or leaking containment', () => {
        const opener = document.createElement('button');
        opener.textContent = 'Strict opener';
        document.body.append(opener);
        opener.focus();

        const view = render(
            <StrictMode>
                <Harness active />
            </StrictMode>,
        );
        expect(screen.getByRole('alertdialog', { name: 'Fatal worker error' })).toHaveFocus();
        expect(screen.getByTestId('background')).toHaveAttribute('inert');

        view.rerender(
            <StrictMode>
                <Harness active={false} />
            </StrictMode>,
        );
        expect(opener).toHaveFocus();
        expect(screen.getByTestId('background')).not.toHaveAttribute('inert');
        expect(screen.getByTestId('background')).not.toHaveAttribute('aria-hidden');

        const later = document.createElement('button');
        document.body.append(later);
        later.focus();
        expect(later).toHaveFocus();

        later.remove();
        opener.remove();
    });
});
