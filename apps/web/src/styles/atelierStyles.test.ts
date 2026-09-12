import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const css = readFileSync(resolve(__dirname,'atelier.css'),'utf8');
const light = css.slice(css.indexOf(':root {'),css.indexOf(':root[data-theme="dark"]'));
const dark = css.slice(css.indexOf(':root[data-theme="dark"]'),css.indexOf('html,body,#root'));
const color = (block: string, token: string) => block.match(new RegExp(`${token}:\\s*(#[\\da-f]{6})`,'i'))?.[1].toUpperCase();
const luminance = (hex: string) => [1,3,5].map((i) => parseInt(hex.slice(i,i+2),16)/255).map((v) => v<=.04045 ? v/12.92 : ((v+.055)/1.055)**2.4).reduce((total,value,i) => total+value*[.2126,.7152,.0722][i],0);
const contrast = (a: string,b: string) => (Math.max(luminance(a),luminance(b))+.05)/(Math.min(luminance(a),luminance(b))+.05);

describe('Signal Atelier design contract', () => {
    it('retains the selected canvas, text, rule and accent palette in both themes', () => {
        for (const [token,a,b] of [
            ['--bg-primary','#F7F6F2','#17191B'],['--text-primary','#202225','#EEEDE8'],
            ['--text-secondary','#666970','#BCC0C2'],['--border-color','#D7D7D2','#43484D'],
            ['--color-primary','#D64C28','#EF653F'],['--action-bg','#C74424','#EF653F'],
        ]) {expect(color(light,token)).toBe(a);expect(color(dark,token)).toBe(b);}
    });
    it('keeps normal supporting text and action text above AA contrast in both themes', () => {
        for (const block of [light,dark]) {
            for (const token of ['--text-primary','--text-secondary']) expect(contrast(color(block,token)!,color(block,'--bg-primary')!)).toBeGreaterThanOrEqual(4.5);
            expect(contrast(color(block,'--action-text')!,color(block,'--action-bg')!)).toBeGreaterThanOrEqual(4.5);
        }
    });
    it('uses one canonical token sheet and excludes retired shell styles from the entry', () => {
        const main = readFileSync(resolve(__dirname,'../main.tsx'),'utf8');
        expect(main).toContain("import './styles/atelier.css'");
        expect(main).toContain("import './styles/components.css'");
        expect(main).not.toMatch(/styles\/(index|forge|precisionLab)\.css/);
        const shared = readFileSync(resolve(__dirname,'components.css'),'utf8');
        expect(shared).not.toMatch(/:root\s*\{/);
        expect(shared).not.toContain('.forge-shell');
    });
});
