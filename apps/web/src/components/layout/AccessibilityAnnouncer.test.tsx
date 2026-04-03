import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AccessibilityAnnouncer } from './AccessibilityAnnouncer';

describe('AccessibilityAnnouncer', () => {
    it('announces training state changes', () => {
        const { rerender } = render(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        rerender(
            <AccessibilityAnnouncer
                status="running"
                dataConfigLoading={false}
                networkConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Training started');

        rerender(
            <AccessibilityAnnouncer
                status="paused"
                dataConfigLoading={false}
                networkConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Training paused');
    });

    it('announces loading and error state changes', () => {
        const { rerender } = render(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        rerender(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={true}
                networkConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Generating data');

        rerender(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                configError="Failed to update configuration"
                configErrorSource="network"
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Network error: Failed to update configuration');
    });
});
