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
                featuresConfigLoading={false}
                trainingConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        rerender(
            <AccessibilityAnnouncer
                status="running"
                dataConfigLoading={false}
                networkConfigLoading={false}
                featuresConfigLoading={false}
                trainingConfigLoading={false}
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
                featuresConfigLoading={false}
                trainingConfigLoading={false}
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
                featuresConfigLoading={false}
                trainingConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        rerender(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={true}
                networkConfigLoading={false}
                featuresConfigLoading={false}
                trainingConfigLoading={false}
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
                featuresConfigLoading={false}
                trainingConfigLoading={false}
                configError="Failed to update configuration"
                configErrorSource="network"
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Network error: Failed to update configuration');
    });

    it('announces feature and training loading state changes', () => {
        const { rerender } = render(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                featuresConfigLoading={false}
                trainingConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        rerender(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                featuresConfigLoading={true}
                trainingConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Updating features');

        rerender(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                featuresConfigLoading={false}
                trainingConfigLoading={true}
                configError={null}
                configErrorSource={null}
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Updating training');
    });

    it('announces feature and training errors', () => {
        const { rerender } = render(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                featuresConfigLoading={false}
                trainingConfigLoading={false}
                configError={null}
                configErrorSource={null}
            />,
        );

        rerender(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                featuresConfigLoading={false}
                trainingConfigLoading={false}
                configError="Failed to update features"
                configErrorSource="features"
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Features error: Failed to update features');

        rerender(
            <AccessibilityAnnouncer
                status="idle"
                dataConfigLoading={false}
                networkConfigLoading={false}
                featuresConfigLoading={false}
                trainingConfigLoading={false}
                configError="Failed to update training"
                configErrorSource="training"
            />,
        );

        expect(screen.getByRole('status')).toHaveTextContent('Training error: Failed to update training');
    });
});
