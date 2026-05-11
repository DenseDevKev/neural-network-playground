import { describe, expect, it } from 'vitest';
import { getTrainingLifecycleUi } from './trainingLifecycle.ts';

describe('getTrainingLifecycleUi', () => {
  it('labels idle, running, and paused controls by lifecycle state', () => {
    expect(getTrainingLifecycleUi({ status: 'idle', pauseReason: null, pendingConfigSource: null })).toMatchObject({
      primaryLabel: 'Start',
      primaryAriaLabel: 'Start training',
      isBlocked: false,
    });

    expect(getTrainingLifecycleUi({ status: 'running', pauseReason: null, pendingConfigSource: null })).toMatchObject({
      primaryLabel: 'Pause',
      primaryAriaLabel: 'Pause training',
      isBlocked: false,
    });

    expect(getTrainingLifecycleUi({ status: 'paused', pauseReason: 'manual', pendingConfigSource: null })).toMatchObject({
      primaryLabel: 'Resume',
      primaryAriaLabel: 'Resume training',
      statusText: 'Paused manually',
    });
  });

  it('prioritizes config-blocked copy when a sync is pending', () => {
    expect(getTrainingLifecycleUi({
      status: 'paused',
      pauseReason: 'diverged',
      pendingConfigSource: 'preset',
    })).toMatchObject({
      primaryLabel: 'Resume',
      disabledReason: 'Updating preset config...',
      statusText: 'Updating preset config...',
      isBlocked: true,
    });
  });
});
