import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { Dialog } from './ui';

const showModalDescriptor = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'showModal');
const closeDescriptor = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, 'close');

afterEach(() => {
    if (showModalDescriptor) Object.defineProperty(HTMLDialogElement.prototype, 'showModal', showModalDescriptor);
    else delete (HTMLDialogElement.prototype as Partial<HTMLDialogElement>).showModal;
    if (closeDescriptor) Object.defineProperty(HTMLDialogElement.prototype, 'close', closeDescriptor);
    else delete (HTMLDialogElement.prototype as Partial<HTMLDialogElement>).close;
});

describe('Dialog', () => {
    it('closes the native modal while connected before restoring its opener', () => {
        let connectedAtClose: boolean | null = null;
        Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
            configurable: true,
            value(this: HTMLDialogElement) { this.setAttribute('open', ''); },
        });
        Object.defineProperty(HTMLDialogElement.prototype, 'close', {
            configurable: true,
            value(this: HTMLDialogElement) {
                connectedAtClose = this.isConnected;
                this.removeAttribute('open');
            },
        });

        const view = render(<button type="button">Utilities</button>);
        const opener = screen.getByRole('button', { name: 'Utilities' });
        opener.focus();
        view.rerender(<><button type="button">Utilities</button><Dialog title="Guidance" onClose={() => undefined}>Guidance settings</Dialog></>);

        expect(screen.getByRole('dialog', { name: 'Guidance' })).toHaveFocus();
        view.rerender(<button type="button">Utilities</button>);

        expect(connectedAtClose).toBe(true);
        expect(opener).toHaveFocus();
    });
});
