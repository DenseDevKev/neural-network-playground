import { beforeEach, describe, expect, it } from 'vitest';
import { useLayoutStore } from '../store/useLayoutStore.ts';
import { focusExplanationActionTarget } from './explanationActionFocus.ts';

describe('explanation navigation and focus', () => {
    beforeEach(() => {
        localStorage.clear(); document.body.innerHTML = '';
        useLayoutStore.setState({destination:'playground',workspaceTab:'network',setupTab:'dataset',resultsTab:'boundary',inspectTab:'trace',audienceMode:'explore',exportRequest:null});
    });
    it.each([['presets','dataset'],['data','dataset'],['features','network'],['network','network'],['hyperparams','training']] as const)('opens %s in the shared %s draft editor and focuses its selected section', (target,tab) => {
        document.body.innerHTML = '<nav class="atelier-setup-tabs"><button aria-current="page">Selected setup section</button></nav>';
        focusExplanationActionTarget(target,{scheduleFocus:fn=>fn()});
        expect(useLayoutStore.getState()).toMatchObject({destination:'playground',workspaceTab:'setup',setupTab:tab});
        expect(document.activeElement).toBe(document.querySelector('button'));
    });
    it.each([['loss','learning'],['confusion','errors'],['boundary','boundary']] as const)('opens %s in Results and focuses its active tab', (target,tab) => {
        document.body.innerHTML = '<div aria-label="Results views"><button aria-selected="true">Evidence tab</button></div>';
        focusExplanationActionTarget(target,{scheduleFocus:fn=>fn()});
        expect(useLayoutStore.getState()).toMatchObject({workspaceTab:'results',resultsTab:tab});
        expect(document.activeElement).toBe(document.querySelector('button'));
    });
    it('opens Inspect with More guidance without changing explanation density', () => {
        useLayoutStore.setState({audienceMode:'beginner'});
        document.body.innerHTML='<button id="workspace-tab-inspect">Inspect</button>';
        focusExplanationActionTarget('inspection',{scheduleFocus:fn=>fn()});
        expect(useLayoutStore.getState()).toMatchObject({workspaceTab:'inspect',audienceMode:'beginner'});
        expect(document.activeElement).toBe(document.querySelector('button'));
    });
    it.each([['config','setup'],['code','code']] as const)('opens the %s utility without discarding the workspace selection', (target,mode) => {
        focusExplanationActionTarget(target,{scheduleFocus:fn=>fn()});
        expect(useLayoutStore.getState()).toMatchObject({workspaceTab:'network',exportRequest:{mode}});
    });
    it('opens Saved runs for a history action', () => {
        focusExplanationActionTarget('history',{scheduleFocus:fn=>fn()});
        expect(useLayoutStore.getState().destination).toBe('saved-runs');
    });
});
