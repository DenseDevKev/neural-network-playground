import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { FirstVisitLessonCue } from './FirstVisitLessonCue.tsx';

const FRESH_PROPS = {
    historyReady: true,
    lessonCueDismissed: false,
    hasSavedRuns: false,
    hasStartedLesson: false,
    hasActiveLesson: false,
    onOpenLessons: vi.fn(),
    onDismiss: vi.fn(),
};

describe('FirstVisitLessonCue', () => {
    it('offers a short lesson to a fresh user', () => {
        render(<FirstVisitLessonCue {...FRESH_PROPS} />);

        expect(screen.getByRole('region', { name: 'Getting started' })).toBeVisible();
        expect(screen.getByRole('button', { name: 'Start a 3-minute lesson' })).toBeVisible();
    });

    it.each([
        ['history is still loading', { historyReady: false }],
        ['the cue was dismissed', { lessonCueDismissed: true }],
        ['a run was saved', { hasSavedRuns: true }],
        ['a lesson was previously started or finished', { hasStartedLesson: true }],
        ['a lesson is active', { hasActiveLesson: true }],
    ])('stays hidden when %s', (_reason, overrides) => {
        render(<FirstVisitLessonCue {...FRESH_PROPS} {...overrides} />);

        expect(screen.queryByRole('region', { name: 'Getting started' }))
            .not.toBeInTheDocument();
    });

    it('routes its actions through the supplied open and dismiss callbacks', async () => {
        const user = userEvent.setup();
        const onOpenLessons = vi.fn();
        const onDismiss = vi.fn();
        render(
            <FirstVisitLessonCue
                {...FRESH_PROPS}
                onOpenLessons={onOpenLessons}
                onDismiss={onDismiss}
            />,
        );

        await user.click(screen.getByRole('button', { name: 'Start a 3-minute lesson' }));
        expect(onOpenLessons).toHaveBeenCalledTimes(1);
        expect(onDismiss).not.toHaveBeenCalled();

        await user.click(screen.getByRole('button', { name: 'Dismiss lesson suggestion' }));
        expect(onDismiss).toHaveBeenCalledTimes(1);
    });
});
