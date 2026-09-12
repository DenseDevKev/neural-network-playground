import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import { useState } from 'react';
import { DEFAULT_EXPERIMENT_DOCUMENT } from '@nn-playground/shared';
import { useRecipeDraft, type SetupTab } from '../../../hooks/useRecipeDraft.ts';
import { usePlaygroundStore } from '../../../store/usePlaygroundStore.ts';
import { useTrainingStore } from '../../../store/useTrainingStore.ts';
import { SetupEditor } from './SetupEditor.tsx';
function Harness() { const controller=useRecipeDraft();const [tab,onTabChange]=useState<SetupTab>('dataset');return <SetupEditor {...{controller,tab,onTabChange}}/>; }
beforeEach(async () => {await usePlaygroundStore.getState().replaceDocument(DEFAULT_EXPERIMENT_DOCUMENT);useTrainingStore.getState().finishConfigChange();useTrainingStore.setState({configError:null,configErrorSource:null});});
it('retains one draft across tabs and cancels every pending section', () => {
    render(<Harness/>);
    fireEvent.change(screen.getByLabelText('Samples'),{target:{value:'100'}});
    fireEvent.click(screen.getByRole('button',{name:'Training'}));
    fireEvent.change(screen.getByLabelText('Batch size'),{target:{value:'200'}});
    expect(screen.getByRole('button',{name:'Apply setup'})).toBeDisabled();
    fireEvent.click(screen.getByRole('button',{name:'Data'}));
    expect(screen.getByLabelText('Samples')).toHaveValue('100');
    fireEvent.change(screen.getByLabelText('Samples'),{target:{value:'1000'}});
    expect(screen.getByRole('button',{name:'Apply setup'})).toBeEnabled();
    fireEvent.click(screen.getByRole('button',{name:'Cancel'}));
    expect(screen.getByLabelText('Samples')).toHaveValue(String(DEFAULT_EXPERIMENT_DOCUMENT.recipe.data.sampleCount));
    fireEvent.click(screen.getByRole('button',{name:'Training'}));
    expect(screen.getByLabelText('Batch size')).toHaveValue(String(DEFAULT_EXPERIMENT_DOCUMENT.recipe.training.batchSize));
});
