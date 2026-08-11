import { useEffect, type RefObject } from 'react';

interface AttributeSnapshot {
    hadAttribute: boolean;
    value: string | null;
}

interface BackgroundLease {
    element: HTMLElement;
    inert: AttributeSnapshot;
    ariaHidden: AttributeSnapshot;
}

const FOCUSABLE_SELECTOR = [
    'a[href]',
    'area[href]',
    'button',
    'input',
    'select',
    'textarea',
    '[contenteditable]:not([contenteditable="false"])',
    '[tabindex]',
].join(',');

export function useModalFocusContainment(
    active: boolean,
    dialogRef: RefObject<HTMLElement | null>,
    backgroundRef: RefObject<HTMLElement | null>,
) {
    useEffect(() => {
        if (!active || typeof document === 'undefined') return;

        const dialog = dialogRef.current;
        const background = backgroundRef.current;
        if (!dialog || !background) return;

        const activeElement = document.activeElement;
        const priorFocus = activeElement instanceof HTMLElement && !dialog.contains(activeElement)
            ? activeElement
            : null;
        const leases = new Map<HTMLElement, BackgroundLease>();

        const attributeSnapshot = (element: HTMLElement, name: string): AttributeSnapshot => ({
            hadAttribute: element.hasAttribute(name),
            value: element.getAttribute(name),
        });

        const lease = (element: HTMLElement) => {
            if (leases.has(element) || element === dialog || element.contains(dialog)) return;

            leases.set(element, {
                element,
                inert: attributeSnapshot(element, 'inert'),
                ariaHidden: attributeSnapshot(element, 'aria-hidden'),
            });
            element.setAttribute('inert', '');
            element.setAttribute('aria-hidden', 'true');
        };

        const leaseBackground = () => {
            lease(background);
            for (const child of document.body.children) {
                if (child instanceof HTMLElement && !child.contains(dialog)) {
                    lease(child);
                }
            }
        };

        const isNativeContentEditable = (element: HTMLElement) => (
            element.matches('[contenteditable]:not([contenteditable="false"])')
        );

        const isSequentiallyFocusable = (element: HTMLElement) => {
            const hasExplicitNegativeTabIndex = element.hasAttribute('tabindex')
                && element.tabIndex < 0;
            return !hasExplicitNegativeTabIndex
                && (element.tabIndex >= 0 || isNativeContentEditable(element));
        };

        const hasHiddenContext = (element: HTMLElement, boundary?: HTMLElement) => {
            let current: HTMLElement | null = element;
            while (current) {
                if (
                    current.hidden
                    || current.hasAttribute('inert')
                    || current.getAttribute('aria-hidden') === 'true'
                ) {
                    return true;
                }
                const style = window.getComputedStyle(current);
                if (
                    style.display === 'none'
                    || style.visibility === 'hidden'
                    || style.visibility === 'collapse'
                ) {
                    return true;
                }
                if (current === boundary) break;
                current = current.parentElement;
            }
            return false;
        };

        const isHiddenOrDisabled = (element: HTMLElement) => (
            element.matches(':disabled')
            || !isSequentiallyFocusable(element)
            || hasHiddenContext(element, dialog)
        );

        const focusables = () => Array.from(
            dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
        ).filter((element) => !isHiddenOrDisabled(element));

        const focusWithoutScroll = (element: HTMLElement) => {
            try {
                element.focus({ preventScroll: true });
            } catch {
                element.focus();
            }
        };

        focusWithoutScroll(dialog);
        leaseBackground();

        const observer = new MutationObserver(() => leaseBackground());
        observer.observe(document.body, { childList: true });

        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Tab') {
                event.preventDefault();
                event.stopPropagation();

                const candidates = focusables();
                if (candidates.length === 0) {
                    focusWithoutScroll(dialog);
                    return;
                }

                const currentIndex = candidates.indexOf(document.activeElement as HTMLElement);
                const nextIndex = event.shiftKey
                    ? (currentIndex <= 0 ? candidates.length - 1 : currentIndex - 1)
                    : (currentIndex < 0 || currentIndex === candidates.length - 1 ? 0 : currentIndex + 1);
                focusWithoutScroll(candidates[nextIndex]);
                return;
            }

            if (event.key === 'Escape') {
                event.preventDefault();
                event.stopPropagation();
            }
        };

        const onFocusIn = (event: FocusEvent) => {
            if (event.target instanceof Node && dialog.contains(event.target)) return;
            focusWithoutScroll(dialog);
        };

        document.addEventListener('keydown', onKeyDown, true);
        document.addEventListener('focusin', onFocusIn, true);

        return () => {
            observer.disconnect();
            document.removeEventListener('keydown', onKeyDown, true);
            document.removeEventListener('focusin', onFocusIn, true);

            for (const { element, inert, ariaHidden } of Array.from(leases.values()).reverse()) {
                if (inert.hadAttribute) {
                    element.setAttribute('inert', inert.value ?? '');
                } else {
                    element.removeAttribute('inert');
                }
                if (ariaHidden.hadAttribute) {
                    element.setAttribute('aria-hidden', ariaHidden.value ?? '');
                } else {
                    element.removeAttribute('aria-hidden');
                }
            }

            if (
                !priorFocus?.isConnected
                || priorFocus.matches(':disabled')
                || !isSequentiallyFocusable(priorFocus)
                || hasHiddenContext(priorFocus)
            ) {
                return;
            }
            focusWithoutScroll(priorFocus);
        };
    }, [active, backgroundRef, dialogRef]);
}
